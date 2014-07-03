#!/usr/bin/python
import os
import emcee
from emcee.utils import MPIPool
from mpi4py import MPI
import numpy
import phoebe
from phoebe.backend import universe
import pickle
import sys
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def save_pickle(data, fn):
    f = open(fn, 'w')
    pickle.dump(data, f)
    f.close()

def lnprob(values, pars, system, compute_params):
    
    # Make sure there are no previous calculations left
    system.reset_and_clear()
    
    # Set the new values of the parameters, report the rank of this walker
    rank = MPI.COMM_WORLD.Get_rank()
    for par, value in zip(pars, values):
        par.set_value(value)
    
    # Evaluate the system, get the results and return a probability
    had = []
    
    # Walk through all the parameterSets available. Collect unique
    # parameters and their values, but stop once a parameter value is
    # outside it's limits or prior boundaries (raise StopIteration). In the
    # latter case, we don't need to compute the model anymore, but can just
    # return logp=-np.inf
    try:
        for parset in system.walk():
            
            # For each parameterSet, walk through all the parameters
            for qual in parset:
                
                # Don't bother if the parameter has no prior
                if not parset.has_prior(qual):
                    continue
                
                # Ask a unique ID and update the value of the parameter
                this_param = parset.get_parameter(qual)
                myid = this_param.get_unique_label()
                
                # If we already incountered this parameter, continue
                # on to the next
                if myid in had:
                    continue
                    
                had.append(myid)
                    
                # If this parameter is outside the limits, we know the
                # model is crap and forget about it immediately
                if not this_param.is_inside_limits():
                    raise StopIteration
                    
                # If this parameter is outside the boundaries of the
                # prior, the model is crap and forget about it
                # immediately
                if np.isinf(this_param.get_logp()):
                    raise StopIteration
                                  
    # If any of the parameters is outside the bounds, we don't really
    # compute the model
    except StopIteration:
        return -np.inf
    
    try:
        system.compute(params=compute_params)
    except Exception as msg:
        print("Compute failed with message: {} --> logp=-inf".format(str(msg)))
        return -np.inf
    
    logp, chi2, N = system.get_logp(include_priors=True)
    # Remember the result from the simulations
    #obs = system.get_obs()
    #return logp, (obs['scale'], obs['offset'])

    return logp
    
    
    


def rpars(mysystem, nwalkers):
    
    # What parameters need to be adjusted?
    pars = mysystem.get_adjustable_parameters()
    
    # Create an initial set of parameters from the priors
    p0 = [par.get_value_from_prior(size=nwalkers) for par in pars]
    p0 = np.array(p0).T # Nwalkers x Npar
    
    # we do need to check if all the combinations produce realistic models
    for i, walker in enumerate(p0):
                    
        # Check the model for this set of parameters
        while True:
            
            # Set the values
            for par, value in zip(pars, walker):
                par.set_value(value)
            
            # If it checks out, continue checking the next one
            if not any([np.isinf(par.get_logp()) for par in pars]):
                p0[i] = walker
                break
            
            # Else draw a new value
            walker = [par.get_value_from_prior(size=1)[0] for par in pars]
    return p0



def multivariate_priors(filename, start, lnproblim, sigmlt=1.0, nnwalkers=0):
        """
        Generate a new prior distribution using a subset of walkers
        with lnprob > ``lnproblim``.

        :param filename:
            Filename of the chain file.

        :param start:
            Starting point in the chain.

        :param lnproblim:
            Limiting log probability.

        :param sigmlt:
            Rescale original sigmas for this much.

        :param nnwalkers:
            New number of walkers. If 0, use the old value.

        """
        d = np.loadtxt(filename)

        nwalkers = len(set(d[:, 0]))
        npars = len(d[0]) - 1

        if start * nwalkers > len(d):
            print '`start` is too far ahead...'
            return None

        logp = d[:, -1][start * nwalkers:]

        averages = [np.average(d[:, i][start * nwalkers:][logp > lnproblim])
                    for i in range(1, npars)]
        sigmas = np.array([np.std(d[:, i][start * nwalkers:][logp > lnproblim])
                           for i in range(1, npars)]) * sigmlt

        cor = np.zeros((npars - 1, npars - 1))
        for i in range(1, npars):
            for j in range(1, npars):
                prs = st.pearsonr(d[:, i][start * nwalkers:][logp > lnproblim],
                                  d[:, j][start * nwalkers:][logp > lnproblim])[0]
                cor[i - 1][j - 1] = prs * sigmas[i - 1] * sigmas[j - 1]

        #~ smpls = np.random.multivariate_normal(averages, cor, nwalkers)
        return np.random.multivariate_normal(averages, cor, nwalkers)

    


def update_progress(system, sampler, fitparams, last=10):
    adj_pars = system.get_adjustable_parameters()
    k = sum(sampler.lnprobability[0]<0)
    chain = sampler.chain[:,:k,:]
    logprobs = sampler.lnprobability[:,:k]
    best_iter = np.argmax(logprobs.max(axis=0))
    best_walker = np.argmax(logprobs[:,best_iter])
    text = ["EMCEE Iteration {:>6d}/{:<6d}: {} walkers, {} parameters".format(k-1, fitparams['iters'], fitparams['walkers'], chain.shape[-1])]
    text+= ["      best logp = {:.6g} (reached at iter {}, walker {})".format(logprobs.max(axis=0)[best_iter],best_iter, best_walker)]
    for i, par in enumerate(adj_pars):
        pos = chain[:,-last:,i].ravel()
        text.append("  {:15s} = {:.6g} +/- {:.6g} (best={:.6g})".format(par.get_qualifier(),
                                     np.median(pos), np.std(pos), chain[best_walker, best_iter, i]))
    
    text = "\n".join(text) +'\n'
    print(text)
    
    




def run(system_file, compute_params_file, fit_params_file, state=None):
    
    # Take care of the MPI pool
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    
    # Load the System
    system = universe.load(system_file)
    system.reset_and_clear()
    
    # Load the fit and compute ParameterSets
    compute_params = universe.load(compute_params_file)
    fit_params = universe.load(fit_params_file)
    
    
    # Retrieve the number of walkers and iterations
    nwalkers = fit_params['walkers']
    niters = fit_params['iters']
    label = fit_params['label']
    
    # We need unique names for the parameters that need to be fitted, we need
    # initial values and identifiers to distinguish parameters with the same
    # name (we'll also use the identifier in the parameter name to make sure
    # they are unique). While we are iterating over all the parameterSets,
    # we'll also have a look at what context they are in. From this, we decide
    # which fitting algorithm to use.
    ids = []
    pars = []
    names = []
    pars_objects = []
    
    # walk through all the parameterSets available. This needs to be via this
    # utility function because we need to iteratively walk down through all
    # BodyBags too.
    frames = []
    for parset in system.walk():
        frames.append(parset.frame)
        
        # for each parameterSet, walk through all the parameters
        for qual in parset:
            
            # extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                
                # ask a unique ID and check if this parameter has already been
                # treated. If so, continue to the next one.
                parameter = parset.get_parameter(qual)
                myid = parameter.get_unique_label()
                if myid in ids:
                    continue
                
                # and add the id
                ids.append(myid)
                pars.append(parameter.get_value_from_prior(size=nwalkers))
                names.append(qual)
                pars_objects.append(parameter)
    
    pars = np.array(pars).T
    ndim = len(ids)
    
    # now, if the number of walkers is smaller then twice the number of
    # parameters, adjust that number to the required minimum and raise a warning
    
    if (2*pars.shape[1]) > nwalkers:
        logger.warning("Number of walkers ({}) cannot be smaller than 2 x npars: set to {}".format(nwalkers,2*pars.shape[1]))
        nwalkers = 2*pars.shape[1]
        fit_params['walkers'] = nwalkers
    
    # Initialize a set of parameters
    p0 = rpars(system, nwalkers)
        
    # Create the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                  args=[pars_objects, system, compute_params], pool=pool)
    
    # And run!
    f = open(label + '.mcmc_chain.dat', "w")
    f.write("# walker " + " ".join(ids) + " logp\n")
    f.close()
    
    generator = sampler.sample(p0, iterations=niters, storechain=True)
    for niter, result in enumerate(generator):
        #print("Iteration {}".format(niter))
        position, logprob, rstate  = result
        
        # Write walker positions
        with open(label + '.mcmc_chain.dat', "a") as f:
            for k in range(position.shape[0]):
                f.write("%d %s %f\n"
                        % (k, " ".join(['%.10f' % i for i in position[k]]), logprob[k]))
            
                ## Write results of the simulations
                #if blobs[k] is not None:
                    #blobs[k].save('simuls/' + phoebe_file + '.simul_{:05d}_{:05d}.lc'.format(niter, k),
                          #pretty_header=True)
        if niter<10:
            update_progress(system, sampler, fit_params)
        elif niter<50 and niter%2 == 0:
            update_progress(system, sampler, fit_params)
        elif niter<500 and niter%10 == 0:
            update_progress(system, sampler, fit_params)
        elif niter%100 == 0:
            update_progress(system, sampler, fit_params)    
            
            #save_pickle([result, 0, sampler.chain], phoebe_file + '.mcmc_run.dat')
        
    pool.close()
    save_pickle([result, 0, sampler.chain], label + '.mcmc_run.dat')



if __name__ == '__main__':
    
    system_file = sys.argv[1]
    fit_params_file = sys.argv[2]
    compute_params_file = sys.argv[3]
    run(system_file, fit_params_file, compute_params_file)
    
    
