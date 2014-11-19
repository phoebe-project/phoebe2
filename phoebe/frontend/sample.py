#!/usr/bin/python
import os
import logging
import pickle
import sys
import json
import time
import numpy as np
#from phoebe.fontend.bundle import Bundle
from phoebe.frontend.common import _xy_from_category
from phoebe.utils import utils
try:
    from mpi4py import MPI
except ImportError:
    mpi = False
else:
    mpi = True
    
logger = logging.getLogger("FRONT.SAMPLE")

TAG_REQ = 99
TAG_DATA = 88
TAG_RES  = 77
TAG_COMP = 66
MAX_BYTESIZE = 1000000000

def load_pickle(fn):
    ff = open(fn, 'r')
    myclass = pickle.load(ff)
    ff.close()
    return myclass
    
def single_sample(info):
    """
    """
    
    # get info
    bundle_file             = info.get('bundle_file')
    compute_params_file     = info.get('compute_params_file')
    objref                  = info.get('objref')
    param_values            = info.get('param_values')
    synthetic_twigs         = info.get('synthetic_twigs')
    synthetic_yks           = info.get('synthetic_yks')
    
    # load the bundle and computeoptions from their pickled files
    b = load_pickle(bundle_file)
    computeoptions = load_pickle(compute_params_file)
    obj = b.get_object(objref)
    
    # update values 
    for twig, value in param_values.items():
        b.set_value(twig, value)
        
    if not b.check():
        # then these parameters will fail
        return (None, None)
        
    # run compute and keep result
    if computeoptions['time'] == 'auto':
        obj.compute(**computeoptions)
    else:
        raise ValueError("time must be set to 'auto' in compute options")    

    # now let's get the synthetic arrays and return them
    syn_xs = {}
    syn_ys = {}
    for twig, yk in zip(synthetic_twigs, synthetic_yks):
        syn_xs[twig] = b.get(twig, hidden=True)['time']
        syn_ys[twig] = b.get(twig, hidden=True)[yk]
     
    return (syn_xs, syn_ys)
        

def worker(comm, myrank):
    while True:
        # tell the master I'm ready
        comm.send (myrank, 0, tag=TAG_REQ)
        # receive a job from the master
        info = comm.recv (tag=TAG_DATA)

        if not info['continue']:
            comm.send (myrank, 0, tag=TAG_COMP)
            break

        resp = single_sample(info)
        comm.send(resp, 0, tag=TAG_RES)
        
    return

def run(bundle_file, compute_params_file, objref, sample_from, samples, avg=True):
    """
    
    """
    if mpi:
        comm = MPI.COMM_WORLD
        myrank = comm.Get_rank()
        nprocs = comm.Get_size()
        
        if myrank!=0:
            # then worker
            # this will cause the worker to enter an infinite loop 
            # until we force it to break, at which point we'll return
            worker(comm, myrank)
            return
            
        resp = []
    else:
        nprocs = 0

    # if we've made it this far, this is the master process
    computeoptions = load_pickle(compute_params_file)
    b = load_pickle(bundle_file)
    
    # determine the parameters we need to change and sample their values
    adjustable_twigs = b.get_adjustable_parameters().keys()
    pars = [b.get_parameter(twig) for twig in adjustable_twigs]
    
    # Draw function: if it's from posteriors and a par has no posterior, fall
    # back to prior
    if sample_from in ['posterior', 'posteriors', 'post']:
        draw_funcs = ['get_value_from_posterior' if par.has_posterior() \
                                    else 'get_value_from_prior' for par in pars]
        get_funcs = ['get_posterior' if par.has_posterior() \
                                    else 'get_prior' for par in pars]
    elif sample_from in ['priors', 'prior']:
        draw_funcs = ['get_value_from_prior' for par in pars]
        get_funcs = ['get_prior' for par in pars]
    else:
        raise ValueError("sample_from must be one of: 'prior', 'posterior'")
    
    #~ p0 = [getattr(par, draw_func)(size=samples) for par, draw_func in zip(pars, draw_funcs)]
    #~ param_values_per_iter = np.array(p0).T 
    param_values_per_iter = [{twig: getattr(par, draw_func)(size=1)[0] for twig, par, draw_func in zip(adjustable_twigs, pars, draw_funcs)} for i in range(samples)]
    
    # retrieve the list of twigs for datasets we need to track
    synthetic_twigs = b.twigs(class_name='*DataSet', context='*syn', hidden=True)
    
    synthetic_yks = []
    remove_twigs = []
    for twig in synthetic_twigs:
        synds_context = b.get(twig, hidden=True).context[:-3]
        
        if synds_context != 'orb':
            # WE DO NOT CURRENTLY TREAT ORB DATASETS AS COMPUTABLES
            # but this may change in the future
            synthetic_yks.append(_xy_from_category(synds_context)[1])
        else:
            remove_twigs.append(twig)
    
    for twig in remove_twigs:
        synthetic_twigs.remove(twig)
        
    syn_xs = {}
    syn_ys = []
        
    for i,param_values in enumerate(param_values_per_iter):
        info = {'continue': True,
                'param_values': param_values, 
                'bundle_file': bundle_file,
                'compute_params_file': compute_params_file,
                'objref': objref,
                'synthetic_twigs': synthetic_twigs,
                'synthetic_yks': synthetic_yks}
    
        logger.warning("sample #{}: {}".format(i+1, info['param_values']))
    
        if mpi:
            node = comm.recv (source=MPI.ANY_SOURCE, tag=TAG_REQ)
            comm.send (info, node, tag=TAG_DATA)
            
            resp.append(comm.irecv(bytearray(MAX_BYTESIZE),node,tag=TAG_RES))

        else:
            x, y = single_sample(info)
            
            if x is not None and y is not None:
                syn_ys.append(y)
                if not len(syn_xs):
                    syn_xs = x
            
    if nprocs:
        # then we need to wait for the responses
        for i in range(samples):
            x, y = resp[i].wait()

            if x is not None and y is not None:
                syn_ys.append(y)
                if not len(syn_xs):
                    syn_xs = x

        # shutdown the workers
        for node in range(1, nprocs):
            info = {'continue': False}
            comm.send(info, node, tag=TAG_DATA)

    # process the synthetics
    # syn_xs is a dictionary (since all are the same)
    # syn_ys is a list of dictionaries
    data = {'hist': param_values_per_iter, 'syns': {}}
    for twig, yk in zip(synthetic_twigs, synthetic_yks):
        data['syns'][twig] = {}
        
        data['syns'][twig]['x'] = syn_xs[twig]
        data['syns'][twig]['xk'] = 'time'
        
        syn_y = [s[twig] for s in syn_ys]
        syntable = np.array(syn_y).T
        
        if avg:
            # then we want to return xs, ys (avg), sigma (std of the ys at each x)
            data['syns'][twig]['y'] = [np.average(i) for i in syntable]
            data['syns'][twig]['yk'] = yk
            data['syns'][twig]['sigma'] = [np.std(i) for i in syntable]
        else:
            # then we want to return xs, ys (list per x)
            data['syns'][twig]['y'] = syntable  # need to make this serializable
            data['syns'][twig]['yk'] = yk
    
    fname = computeoptions['label'] + '.sample.dat'
    f = open(fname, 'w')
    f.write(json.dumps(data))
    f.close()
    
    return fname


if __name__ == '__main__':
    logger = utils.get_basic_logger(clevel='INFO')
    
    bundle_file = sys.argv[1]
    compute_params_file = sys.argv[2]
    objref = sys.argv[3]
    sample_from = sys.argv[4]
    samples = int(float(sys.argv[5]))
    logger_level = sys.argv[6]
    
    logger.setLevel(logger_level)

    run(bundle_file, compute_params_file, objref, sample_from, samples)
