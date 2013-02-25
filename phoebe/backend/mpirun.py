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
    
    
    # Load the system that we want to compute
    system = universe.load(sys.argv[2])
    # Load the arguments to the function
    with open(sys.argv[3],'r') as ff: args = cPickle.load(ff)
    # Load the keyword arguments to the function
    with open(sys.argv[4],'r') as ff: kwargs = cPickle.load(ff)
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
    system.set_time(params['time'][0])
    
    
    # Instead of computing one phase per worker, we'll let each worker do
    # a few. We're conservative on the number of phases each worker should do,
    # just to make sure that that if one worker is a bit lazy, the rest doesn't
    # have to wait for too long...: e.g. if there are 8 phases left and there
    # are 4 workers, we only take 1 phase point for each worker.
    olength = len(dates)
    while len(dates):
        #-- print some diagnostics to the user.
        print("MPIrun: starting at {:.3f}% of total run".format((olength-len(dates))/float(olength)*100))
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
