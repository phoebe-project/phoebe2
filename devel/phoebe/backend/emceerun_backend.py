#!/usr/bin/python
import os
import pickle
import sys
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
try:
    import emcee
    from emcee.utils import MPIPool
    from mpi4py import MPI
except ImportError:
    pass
from phoebe.backend import universe
from phoebe.utils import utils
mpi = True

logger = utils.get_basic_logger(clevel='INFO')

def save_pickle(data, fn):
    f = open(fn, 'w')
    pickle.dump(data, f)
    f.close()

def lnprob(values, pars, system, compute_params):
    
    # Make sure there are no previous calculations left
    system.reset_and_clear()
    
    # Set the new values of the parameters, report the rank of this walker
    for par, value in zip(pars, values):
        par.set_value(value)
    
    # Walk through all the parameterSets available. Collect unique
    # parameters and their values, but stop once a parameter value is
    # outside it's limits or prior boundaries (raise StopIteration). In the
    # latter case, we don't need to compute the model anymore, but can just
    # return logp=-np.inf
    if not system.check():
        return -np.inf, None
    
    try:
        system.compute(params=compute_params)
    except Exception as msg:
        logger.warning("Compute failed with message: {} --> logp=-inf".format(str(msg)))
        return -np.inf, None
    
    logp, chi2, N = system.get_logp(include_priors=True)
    
    # Get also the parameters that are adjustable but have no prior        
    auto_fitted = system.get_adjustable_parameters(with_priors=False)
    auto_fitted = [par.get_value() for par in auto_fitted]
    #obs = system.get_obs()
    #return logp, (obs['scale'], obs['offset'])
    
    # And finally get the parameters that are not adjustable but have a prior
    derived = system.get_parameters_with_priors(is_adjust=False, is_derived=True)
    derived = [par.get_value() for par in derived]
    
    return logp, (auto_fitted, derived)
    
    
def univariate_init(mysystem, nwalkers, draw_from='prior'):
    
    # What parameters need to be adjusted?
    pars = mysystem.get_adjustable_parameters()
    
    # Draw function
    draw_func = 'get_value_from_' + draw_from
    
    # Create an initial set of parameters from the priors
    p0 = [getattr(par, draw_func)(size=nwalkers) for par in pars]
    p0 = np.array(p0).T # Nwalkers x Npar
    
    
    # we do need to check if all the combinations produce realistic models
    for i, walker in enumerate(p0):
        max_try = 100
        current_try = 0
        # Check the model for this set of parameters
        while True:
            
            # Set the values
            for par, value in zip(pars, walker):
                par.set_value(value)
            
            # If it checks out, continue checking the next one
            #if not any([np.isinf(par.get_logp()) for par in pars]):
            if mysystem.check() or current_try>max_try:
                p0[i] = walker
                break
            
            current_try += 1
            
            # Else draw a new value: for traces, we remember the index so that
            # we can take the parameters from the same representation always
            index = None
            walker = []
            for par in pars:
                distr = getattr(par, 'get_' + draw_from)().get_distribution()[0]
                if distr == 'trace' and index is None:
                    value, index = getattr(par, draw_func)(size=1)
                    value, index = value[0], index[0]
                elif distr == 'trace':
                    trace = getattr(par, 'get_' + draw_from)().get_distribution()[1]['trace'][index]
                    value = trace[index]
                # For any other distribution we simply draw
                else:
                    value = getattr(par, draw_func)(size=1)[0]
                    
                walker.append(value)
    return p0



def multivariate_init(mysystem, nwalkers, draw_from='prior'):
    """
    Generate a new prior distribution using a subset of walkers

    :param nwalkers:
        New number of walkers.

    """
    # What parameters need to be adjusted?
    pars = mysystem.get_adjustable_parameters()
    npars = len(pars)
    
    # Draw function
    draw_func = 'get_value_from_' + draw_from
    
    # Getter
    get_func = 'get_' + draw_from
    
    # Check if distributions are traces, otherwise we can't generate
    # multivariate distributions
    for par in pars:
        this_dist = getattr(par, get_func)().get_distribution()[0]
        if not this_dist == 'trace':
            raise ValueError(("Only trace distributions can be used to "
                              "generate multivariate walkers ({} "
                              "distribution given for parameter "
                              "{})").format(this_dist, par.get_qualifier()))
    
    # Extract averages and sigmas
    averages = [getattr(par, get_func)().get_loc() for par in pars]
    sigmas = [getattr(par, get_func)().get_scale() for par in pars]
    
    # Set correlation coefficients
    cor = np.zeros((npars, npars))
    for i, ipar in enumerate(pars):
        for j, jpar in enumerate(pars):
            prs = st.pearsonr(getattr(ipar, get_func)().distr_pars['trace'],
                              getattr(jpar, get_func)().distr_pars['trace'])[0]
            cor[i, j] = prs * sigmas[i] * sigmas[j]

    # Sample is shape nwalkers x npars
    sample = np.random.multivariate_normal(averages, cor, nwalkers)
    
    # Check if all initial values satisfy the limits and priors. If not,
    # draw a new random sample and check again. Don't try more than 100 times,
    # after that we just need to live with walkers that have zero probability
    # to start with...
    for i, walker in enumerate(sample):
        max_try = 100
        current_try = 0
        while True:
            # Adopt the values in the system
            for par, value in zip(pars, walker):
                par.set_value(value)
                sample[i] = walker
            # Perform the check; if it doesn't work out, retry
            if not mysystem.check() and current_try < max_try:
                walker = np.random.multivariate_normal(averages, cor, 1)[0]
                current_try += 1
            else:
                break
        else:
            logger.warning("Walker {} could not be initalised with valid parameters".format(i))
    
    return sample

    


def update_progress(system, sampler, fitparams, last=10):
    """
    Report the current status of the sampler.
    """
    adj_pars = system.get_adjustable_parameters(with_priors=True)
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
    logger.warning(text)
    
    




def run(system_file, compute_params_file, fit_params_file):
    
    # Take care of the MPI pool
    if mpi:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool = None
            
    # Load the System
    system = universe.load(system_file)
    system.reset_and_clear()
    
    # Load the fit and compute ParameterSets
    compute_params = universe.load(compute_params_file)
    fit_params = universe.load(fit_params_file)
    
    
    # Retrieve the number of walkers and iterations
    nwalkers = fit_params['walkers']
    niters = fit_params['iters']
    label = os.path.join(os.path.dirname(system_file), fit_params['label'])
    
    # We need unique names for the parameters that need to be fitted, we need
    # initial values and identifiers to distinguish parameters with the same
    # name (we'll also use the identifier in the parameter name to make sure
    # they are unique). While we are iterating over all the parameterSets,
    # we'll also have a look at what context they are in. From this, we decide
    # which fitting algorithm to use.
    pars_objects = system.get_adjustable_parameters(with_priors=True)
    ids = [par.get_unique_label() for par in pars_objects]
    names = [par.get_qualifier() for par in pars_objects]
    prior_labels = ["".join(str(par.get_prior()).split()) for par in pars_objects]
    pars = [par.get_value_from_prior(size=nwalkers) for par in pars_objects]
    
    pars = np.array(pars).T
    ndim = len(ids)
    
    # List the parameters that are auto-fitted
    auto = system.get_adjustable_parameters(with_priors=False)
    auto_ids = [par.get_unique_label() for par in auto]
    auto_prior_labels = ['none' for par in auto]
    auto = [par.get_qualifier() for par in auto]
    
    # List the parameters that are derived
    derived = system.get_parameters_with_priors(is_adjust=False, is_derived=True)
    derived_ids = [par.get_unique_label() for par in derived]
    derived_prior_labels = ["".join(str(par.get_prior()).split()) for par in derived]
    derived = [par.get_qualifier() for par in derived]
    
    # Restart from previous state
    existing_file = os.path.isfile(label + '.mcmc_chain.dat')
    if fit_params['incremental'] and existing_file and fit_params['init_from'] == 'previous_run':
        # Load in the data
        existing = np.loadtxt(label + '.mcmc_chain.dat')
        nwalkers = int(existing[:,0].max() + 1)
        niterations = int(existing.shape[0] / nwalkers)
        
        # Reshape in convenient format
        chain = existing[:, 1:1+ndim]
        del existing
        chain = chain.reshape((niterations, nwalkers, ndim))
        
        # Get the start condition
        p0 = chain[-1]
        
        logger.warning("Continuing previous run (starting at iteration {})".format(niterations))
        
    
    # Or start from scratch
    else:
        if fit_params['init_from'] == 'previous_run':
            logger.warning("Cannot continue from previous run, starting new one from priors")
            fit_params['init_from'] = 'prior'
        # now, if the number of walkers is smaller then twice the number of
        # parameters, adjust that number to the required minimum and raise a warning
        
        if (2*pars.shape[1]) > nwalkers:
            logger.warning("Number of walkers ({}) cannot be smaller than 2 x npars: set to {}".format(nwalkers,2*pars.shape[1]))
            nwalkers = 2*pars.shape[1]
            fit_params['walkers'] = nwalkers
    
        # Initialize a set of parameters
        try:
            p0 = multivariate_init(system, nwalkers, draw_from=fit_params['init_from'])
            logger.warning("Initialised walkers from {} with multivariate normals".format(fit_params['init_from']))
        except ValueError:
            p0 = univariate_init(system, nwalkers, draw_from=fit_params['init_from'])
            logger.warning("Initialised walkers from {} with univariate distributions".format(fit_params['init_from']))
        
        # Only start a new file if we do not require incremental changes
        if not fit_params['incremental']:
            f = open(label + '.mcmc_chain.dat', "w")
            f.write("# walker " + " ".join(ids) +" "+ " ".join(auto_ids) +" "+ " ".join(derived_ids)+ " logp\n")
            f.write("# walker " + " ".join(names) +" "+ " ".join(auto) +" "+ " ".join(derived)+ " logp\n")
            f.write("# none " + " ".join(prior_labels) +" "+ " ".join(auto_prior_labels) +" "+ " ".join(derived_prior_labels)+ " logp\n")
            f.write("# WALKER " + "FITTED "*len(ids) + "AUTO "*len(auto) + "DERIVED "*len(derived) + "LOGP\n")
            f.close()
    
    
    # Create the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=[pars_objects, system, compute_params],
                                    pool=pool)
    
    # And run!
    generator = sampler.sample(p0, iterations=niters, storechain=True)
    for niter, result in enumerate(generator):
        #print("Iteration {}".format(niter))
        position, logprob, rstate, blobs  = result
        
        # Write walker positions
        with open(label + '.mcmc_chain.dat', "a") as f:
            for k in range(position.shape[0]):
                # Write walkers and ordinary fitting parameters
                f.write("%d %s"
                        % (k, " ".join(['%.10e' % i for i in position[k]])))
                
                # Write auto-fitted parameters
                if len(auto) and blobs[k] is not None:
                    f.write(" %s"
                            % (" ".join(['%.10e' % i for i in blobs[k][0]])))
                elif len(auto):
                    f.write(" %s"
                            % (" ".join(['nan' for i in auto])))
                
                # Write derived parameters
                if len(derived) and blobs[k] is not None:
                    f.write(" %s"
                            % (" ".join(['%.10e' % i for i in blobs[k][1]])))
                elif len(derived):
                    f.write(" %s"
                            % (" ".join(['nan' for i in derived])))
                
                # Write logprob
                f.write(" %.10e\n" % (logprob[k]))
                
        if niter<10:
            update_progress(system, sampler, fit_params)
        elif niter<50 and niter%2 == 0:
            update_progress(system, sampler, fit_params)
        elif niter<500 and niter%10 == 0:
            update_progress(system, sampler, fit_params)
        elif niter%100 == 0:
            update_progress(system, sampler, fit_params)    
            
            #save_pickle([result, 0, sampler.chain], phoebe_file + '.mcmc_run.dat')
    
    if mpi:
        pool.close()
    #save_pickle([result, 0, sampler.chain], label + '.mcmc_run.dat')



if __name__ == '__main__':
    system_file = sys.argv[1]
    fit_params_file = sys.argv[2]
    compute_params_file = sys.argv[3]
    logger_level = sys.argv[4]
    
    logger.setLevel(logger_level)
    
    run(system_file, fit_params_file, compute_params_file)
    
    
