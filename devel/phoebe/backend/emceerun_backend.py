#!/usr/bin/python
import os
import logging
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

logger = logging.getLogger("EMCEE")

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
    
    # And finally get the parameters that are not adjustable but have a prior
    derived = system.get_parameters_with_priors(is_adjust=False, is_derived=True)
    derived = [par.get_value() for par in derived]
    
    return logp, (auto_fitted, derived)
    
    
def univariate_init(mysystem, nwalkers, draw_from='prior'):
    
    # What parameters need to be adjusted?
    pars = mysystem.get_adjustable_parameters()
    
    # Draw function: if it's from posteriors and a par has no posterior, fall
    # back to prior
    if draw_from == 'posterior':
        draw_funcs = ['get_value_from_posterior' if par.has_posterior() \
                                    else 'get_value_from_prior' for par in pars]
        get_funcs = ['get_posterior' if par.has_posterior() \
                                    else 'get_prior' for par in pars]
    else:
        draw_funcs = ['get_value_from_prior' for par in pars]
        get_funcs = ['get_prior' for par in pars]
    
    # Create an initial set of parameters from the priors
    p0 = [getattr(par, draw_func)(size=nwalkers) for par, draw_func in zip(pars, draw_funcs)]
    p0 = np.array(p0).T # Nwalkers x Npar
    
    
    # we do need to check if all the combinations produce realistic models
    exceed_max_try = 0
    difficult_try = 0
    
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
                if current_try>max_try:
                    exceed_max_try += 1
                elif current_try>50:
                    difficult_try += 1
                p0[i] = walker
                break
            
            current_try += 1
            
            # Else draw a new value: for traces, we remember the index so that
            # we can take the parameters from the same representation always
            index = None
            walker = []
            for ii, par in enumerate(pars):
                distr = getattr(par, get_funcs[ii])().get_distribution()[0]
                if distr == 'trace' and index is None:
                    value, index = getattr(par, draw_funcs[ii])(size=1)
                    value, index = value[0], index[0]
                elif distr == 'trace':
                    trace = getattr(par, get_funcs[ii])().get_distribution()[1]['trace'][index]
                    value = trace[index]
                # For any other distribution we simply draw
                else:
                    value = getattr(par, draw_funcs[ii])(size=1)[0]                    
                walker.append(value)
    
    # Perhaps it was difficult to initialise walkers, warn the user
    if exceed_max_try or difficult_try:
        logger.warning(("Out {} walkers, {} were difficult to initialise, and "
                        "{} were impossible: probably your priors are very "
                        "wide and allow many unphysical combinations of "
                        "parameters.").format(len(p0), difficult_try, exceed_max_try))
    
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
        origin = getattr(par, get_func)()
        if origin is None:
            raise ValueError(("No {} defined for parameter {}, cannot "
                              "initialise "
                              "multivariately").format(draw_from, par.get_qualifier))
        this_dist = origin.get_distribution()[0]
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
    if not k:
        logger.warning("No valid models available (chain shape = {})".format(sampler.chain.shape))
        return None
    chain = sampler.chain[:,:k,:]
    logprobs = sampler.lnprobability[:,:k]
    best_iter = np.argmax(logprobs.max(axis=0))
    best_walker = np.argmax(logprobs[:,best_iter])
    text = ["EMCEE Iteration {:>6d}/{:<6d}: {} walkers, {} parameters".format(k-1, fitparams['iters'], fitparams['walkers'], chain.shape[-1])]
    text+= ["      best logp = {:.6g} (reached at iter {}, walker {})".format(logprobs.max(axis=0)[best_iter],best_iter, best_walker)]
    try:
        acc_frac = sampler.acceptance_fraction
        acor = sampler.acor
    except:
        acc_frac = np.array([np.nan])        
    text+= ["      acceptance_fraction ({}->{} (median {}))".format(acc_frac.min(), acc_frac.max(), np.median(acc_frac))]
    for i, par in enumerate(adj_pars):
        pos = chain[:,-last:,i].ravel()
        text.append("  {:15s} = {:.6g} +/- {:.6g} (best={:.6g})".format(par.get_qualifier(),
                                     np.median(pos), np.std(pos), chain[best_walker, best_iter, i]))
    
    text = "\n".join(text) +'\n'
    logger.warning(text)
    
    




def run(system_file, compute_params_file, fit_params_file):
    """
    Run the emcee algorithm.
    
    Following steps need to be taken:
    
    1. The system and parameters need to be loaded from the pickle file
    2. We need to figure out which parameters need to be included in the chain,
       which ones are automatically fitted and which ones are automatically
       derived
    3. Initiate the walkers. This can be done in three ways:
        - starting from the previous run
        - starting from posteriors
        - starting from priors
       This is governed by the "init_from" parameter, and the list above is also
       the order of priority that is used when the init_from option is invalid:
       if there is no previous run, it will be checked if we can start from
       posteriors. If there are no posteriors, we'll start from the priors.
    4. Create the sampler and run!    
    """
    # Take care of the MPI pool
    if mpi:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool = None
    
    # +---------------------------------------+
    # |   STEP 1: load system and parameters  |
    # +---------------------------------------+
    
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
    
    # +--------------------------------------------+
    # |   STEP 2: get all the relevant parameters  |
    # +--------------------------------------------+
    
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
    
    # +-----------------------------------------+
    # |   STEP 3: setup the initial parameters  |
    # +-----------------------------------------+
    
    # Restart from previous state if asked for and if possible
    existing_file = os.path.isfile(label + '.mcmc_chain.dat')
    logger.warning("Using chain file {}".format(label + '.mcmc_chain.dat'))
    if existing_file and fit_params['init_from'] == 'previous_run':
        # Load in the data
        existing = np.loadtxt(label + '.mcmc_chain.dat')
        nwalkers = int(existing[:,0].max() + 1)
        niterations = int(existing.shape[0] / nwalkers)
        
        # Reshape in convenient format
        chain = existing[:, 1:1+ndim]
        chain = chain.reshape((niterations, nwalkers, ndim))
        
        # Get the start condition
        p0 = chain[-1]
        
        # Get the starting lnprob0's
        lnprob0 = existing[-nwalkers:, -1]
        
        # Get the starting blobs (we need to know the autofitteds and autoderiveds)
        with open(label + '.mcmc_chain.dat', 'r') as open_file:
            while True:
                line = open_file.readline()
                if not line:
                    break
                if line[:8] == '# WALKER':
                    line = np.array(line[1:].strip().split(), str)
                    n_auto = np.sum(line=='AUTO')
                    n_derv = np.sum(line=='DERIVED')
                    break
        blobs0 = list(existing[:, -2-n_auto-n_derv:-2])
        blobs0 = [(entry[:n_auto], entry[n_auto:]) for entry in blobs0]
        
        logger.warning("Continuing previous run (starting at iteration {})".format(niterations))
        
        del existing
    
    # Or start from scratch
    else:
        # if previous_run was requested, if we end up here it was not possible.
        # Therefore we set to start from posteriors
        if fit_params['init_from'] == 'previous_run':
            logger.warning("Cannot continue from previous run, falling back to start from posteriors")
            fit_params['init_from'] = 'posterior'
        
        # now, if the number of walkers is smaller then twice the number of
        # parameters, adjust that number to the required minimum and raise a warning
        
        if (2*pars.shape[1]) > nwalkers:
            logger.warning("Number of walkers ({}) cannot be smaller than 2 x npars: set to {}".format(nwalkers,2*pars.shape[1]))
            nwalkers = 2*pars.shape[1]
            fit_params['walkers'] = nwalkers
    
        # Initialize a set of parameters
        try:
            logger.warning("Attempting multivariate initialisation from {}".format(fit_params['init_from']))
            p0 = multivariate_init(system, nwalkers, draw_from=fit_params['init_from'])
            logger.warning("Initialised walkers from {} with multivariate normals".format(fit_params['init_from']))
        except ValueError:
            logger.warning("Attempting univariate initialisation")
            p0 = univariate_init(system, nwalkers, draw_from=fit_params['init_from'])
            logger.warning("Initialised walkers from {} with univariate distributions".format(fit_params['init_from']))
            
        # We don't know the probability of the initial sample (yet)
        lnprob0 = None
        blobs0 = None
        
    # Only start a new file if we do not require incremental changes, or
    # if we start a new file. This overwrites/removes any existing file
    if not fit_params['incremental'] or not existing_file:
        f = open(label + '.mcmc_chain.dat', "w")
        f.write("# walker " + " ".join(ids) +" "+ " ".join(auto_ids) +" "+ " ".join(derived_ids)+ " acc logp\n")
        f.write("# walker " + " ".join(names) +" "+ " ".join(auto) +" "+ " ".join(derived)+ " acc logp\n")
        f.write("# none " + " ".join(prior_labels) +" "+ " ".join(auto_prior_labels) +" "+ " ".join(derived_prior_labels)+ " acc logp\n")
        f.write("# WALKER " + "FITTED "*len(ids) + "AUTO "*len(auto) + "DERIVED "*len(derived) + "ACC LOGP\n")
        f.close()    
    
    
    
    # +--------------------------------------------+
    # |   STEP 4: create the sampler and run!      |
    # +--------------------------------------------+
    
    # Create the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=[pars_objects, system, compute_params],
                                    pool=pool)
    
    # And run!
    generator = sampler.sample(p0, iterations=niters, storechain=True,
                               lnprob0=lnprob0, blobs0=blobs0)
    
    for niter, result in enumerate(generator):
        
        #print("Iteration {}".format(niter))
        position, logprob, rstate, blobs  = result
        
        try:
            acc_frac = sampler.acceptance_fraction
        except:
            acc_frac = np.array([np.nan]*position.shape[0])  
        
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
                
                # Write acceptance fraction
                f.write(" %.10e" % (acc_frac[k]))
                
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
    logger = utils.get_basic_logger(clevel='INFO')
    
    system_file = sys.argv[1]
    fit_params_file = sys.argv[2]
    compute_params_file = sys.argv[3]
    logger_level = sys.argv[4]
    
    logger.setLevel(logger_level)
    
    run(system_file, fit_params_file, compute_params_file)
    
    
