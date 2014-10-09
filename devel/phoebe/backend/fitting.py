"""
Fitting routines, minimization, optimization.

Section 1. Summary
==================

The top level function you should use is :py:func:`run <run>`. The fitting
algorithm is determined by the context of the passed algorithm ParameterSet.
There is also functionality to reiterate the fit, but starting from a different
set of initial parameters and/or performing Monte Carlo simulations with the
data, or by random sampling the data.

**Main function**

.. autosummary::

   run
   
**Fit iterators**

.. autosummary::

   reinitialize_from_priors
   monte_carlo
   subsets_via_flags
   
**Individual fitting routines**

.. autosummary::

   run_pymc
   run_emcee
   run_lmfit
   run_minuit
   run_grid
   run_genetic

**Helper functions**

.. autosummary::

   accept_fit
   summarize_fit
   check_system

   
Section 2. Details
==================

The basic idea behind fitting Phoebe is the following chain of events, once
you defined a model and attached observations:

1. Given a set of parameters, compute a model.
2. Compute observables to mimic the observations attached to that model.
3. **Evaluate the goodness-of-fit**.
4. If **some criterion** is satisfied, then ``STOP``
5. **Change the values** of the adjustable parameters and return to step 1.

Three remarks are in place to explain the statements in bold:

* **some criterion** generally means that you can either stop computing if you
  found a (what you think) optimal set of parameters, or if you computed enough
  models to satisfactory sample the posterior distributions of the adjustable
  parameters.
* **evaluate the goodness-of-fit** means...
* **change the values** means you have to pick a certain algorithm. You can
  use algorithms that try to descent into a minimum of the parameter space,
  algorithms that just try out a whole predefined set of parameters, or
  algorithms that try to walk through the parameter space to sample the
  posterior distributions, rather than finding an optimal value.


"""
import os
import sys
import time
import logging
import functools
import itertools
import pickle
import copy
import tempfile
import subprocess
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from phoebe.utils import utils
from phoebe.backend import universe
from phoebe.backend import decorators
from phoebe.backend import emceerun_backend
from phoebe.wd import wd
from phoebe.parameters import parameters
try:
    import pymc
except ImportError:
    pass
    #print("Unable to load mcmc fitting routines from pymc: restricted fitting facilities")
try:
    import emcee
    from emcee.utils import MPIPool
except ImportError:
    pass
    #print("Unable to load mcmc fitting routines from emcee: restricted fitting facilities")
try:
    import lmfit
except ImportError:
    pass
    #print("Unable to load nonlinear fitting routines from lmfit: restricted fitting facilities")
try:
    import iminuit
except ImportError:
    pass
    #print("Unable to load MINUIT fitting routines from iminuit: restricted fitting facilities")

    
logger = logging.getLogger("FITTING")
fit_logger_level = 'WARNING'


def run(system, params=None, fitparams=None, mpi=None, accept=False, usercosts=None):
    """
    Run a fitting routine.
    
    The type of algorithm to use is defined by the context of C{fitparams}:
    
        1. :ref:`fitting:pymc <parlabel-phoebe-fitting:pymc>`: Metropolis-Hastings MCMC via pymc
        2. :ref:`fitting:emcee <parlabel-phoebe-fitting:emcee>`: Affine Invariant MCMC via emcee
        3. :ref:`fitting:lmfit <parlabel-phoebe-fitting:lmfit>`: nonlinear optimizers via lmfit
        4. :ref:`fitting:minuit <parlabel-phoebe-fitting:minuit>`: nonlinear optimizers via MINUIT
        5. :ref:`fitting:grid <parlabel-phoebe-fitting:grid>`: grid computations
    
    Some fitting contexts have specific subcontexts that allow better control
    of the algorithms used:
    
        1. :ref:`fitting:lmfit:nelder <parlabel-phoebe-fitting:lmfit:nelder>`: Nelder-Mead simplex method
        2. :ref:`fitting:lmfit:leastsq <parlabel-phoebe-fitting:lmfit:leastsq>`: Levenberg-Marquardt method
    
    The parameters determining how the system is computed (in the sense of
    numerics and customized algorithms for the specific problem) are defined by
    the ParameterSet :ref:`params <parlabel-phoebe-compute>`.
    
    The computations can be threaded if an :ref:`mpi <parlabel-phoebe-mpi>`
    ParameterSet is given.
    
    Naturally, only systems with observations attached to them can be fitted.
    
    Only those parameters that have priors and are set to be adjusted are
    included in the fit.
    
    The logging facilities will be temporarily altered but restored to their
    original state after the fit. Specifically, only messages of level ``WARNING``
    will be shown to the screen, the lower levels will be redirected to log
    files, saved in the current directory. There will be one log file per
    MPI thread.
    
    **Example usage**
    
    Suppose you have a spectrum in file ``myspectrum.dat``,
    and you want to fit a Body of type to it, only varying the temperature. Then
    you would first create the ParameterSets::
    
        >>> starpars = phoebe.ParameterSet('star')
        >>> mesh = phoebe.ParameterSet('mesh:marching')
        >>> spobs, spbdep = phoebe.parse_spectrum_as_lprofile('myspectrum.dat')
    
    Then create the Body and attach the observations::
    
        >>> star = phoebe.Star(starpars, mesh=mesh, pbdep=[spdep], obs=[spobs])
    
    Next, you define which parameters you want to fit. For each parameter, set
    the prior and mark it to be adjust in the fit::
    
        >>> teff = starpars.get_parameter('teff')
        >>> teff.set_prior('uniform', lower=5000., upper=7000.)
        >>> teff.set_adjust(True)
    
    You also need to define the fit algorithm you want to use::
    
        >>> fitparams = phoebe.ParameterSet('fitting:lmfit:nelder')
    
    and you can finally start fitting::
    
        >>> fitting.run(star, fitparams=fitparams, mpi=True, accept=True)
    
    
    
    @param system: the system to fit
    @type system: Body
    @param params: computational parameters
    @type params: ParameterSet of context 'compute'
    @param fitparams: fitting parameters
    @type fitparams: ParameterSet of context 'fitting:xxx'
    @param mpi: parameters for mpi
    @type mpi: ParameterSet of context 'mpi'
    @return: fitting parameters and feedback (context 'fitting:xxx')
    @rtype: ParameterSet
    """
    # Determine the type of fitting algorithm to run. If none is given, the
    # default fitter is LMFIT with leastsq algorithm (Levenberg-Marquardt)
    if fitparams is None:
        fitparams = parameters.ParameterSet(frame='phoebe',
                                            context='fitting:lmfit')
        
    # Zeroth we want to check stuff and give a summary of what we're going to do:
    check_system(system, fitparams)
    
    # First we want to get rid of any loggers that are present; we don't want
    # to follow all the output to the screen during each single computation.
    # Only the warning get through, and this is also how the loggers communicate
    if True:
        mylogger = logging.getLogger("")
        for handler in mylogger.handlers:
            if not hasattr(handler,'baseFilename'):
                if mylogger.level<logging._levelNames['WARNING']:
                    handler.setLevel('WARNING')
        #utils.add_filehandler(mylogger,flevel='INFO',
        #        filename='fitting_{}.log'.format("_".join(time.asctime().split())))
        
    # We need to know how to compute the system (i.e. how many subdivisions,
    # reflections etc...)
    if params is None:
        params = parameters.ParameterSet(context='compute')
        logger.info("No compute parameters given: adding default set")
    
    context_hierarchy = fitparams.get_context().split(':')
    if context_hierarchy[1]=='pymc':
        solver = run_pymc
    elif context_hierarchy[1]=='emcee':
        solver = run_emcee
    elif context_hierarchy[1]=='lmfit':
        solver = run_lmfit
    elif context_hierarchy[1]=='minuit':
        solver = run_minuit
    elif context_hierarchy[1]=='grid':
        solver = run_grid
    else:
        raise NotImplementedError("Fitting context {} is not understood".format(fitparams.context))
    
    #run the fitter a couple of times, perhaps with different
    # starting points, different sampling or different noise.
    
    # Perhaps it doesn't make much sense to iterate a fit, then the 
    # parameter might not be given. Set it by default to 1 in that case:
    iters = fitparams.get('iters',1) if fitparams.get_context() not in ['fitting:emcee'] else 1
    feedbacks = []

    # Cycle over all subsets if required. The system itself (i.e. the
    # observational datasets) is changed *inside* the generator function
    for flag, ref in subsets_via_flags(system, fitparams):
        if flag is not None:
            logger.warning("Selected subset from {} via flag {}".format(ref, flag))
        
        # Iterate fit if required
        for iteration in range(iters):
            logger.warning("Iteration {}/{}".format(iteration+1, iters))
            
            # Do stuff like reinitializing the parametersets with values
            # taken from their prior, or add MC noise to the data
            #reinitialize(system, fitparams)
            monte_carlo(system, fitparams)
            
            # Then solve the current system
            if context_hierarchy[1]=='emcee':
                feedback = solver(system, params=params, mpi=mpi, fitparams=fitparams, usercosts=usercosts)
            else:
                feedback = solver(system, params=params, mpi=mpi, fitparams=fitparams)
            feedbacks.append(feedback)
    
    # Sort feedbacks from worst to best fit 
    feedbacks = sorted(feedbacks)
    
    # Reset the logger to get the info messages again
    mylogger.handlers = mylogger.handlers#[:-1]
    for handler in mylogger.handlers:
        if not hasattr(handler,'baseFilename'):
            if mylogger.level>=logging._levelNames['WARNING']:
                handler.setLevel('INFO')
    logger.info("Reset logger")
    
    # If required, accept the fit of the best feedback (we should do this in
    # the frontend!! --> or the fronted should set accept=False and do its own
    # stuff)
    if accept:
        accept_fit(system, feedbacks[-1])
        system.reset()
        system.clear_synthetic()
        try:
            system.compute(params=params, mpi=mpi)
            logger.warning("System recomputed to match best fit")
        except Exception as msg:
            print(system.params.values()[0])
            logger.info("Could not accept for some reason (original message: {})".format(msg))
    
    if len(feedbacks)>1:
        return feedbacks
    else:
        return feedbacks[0]
    
    

                
            
#{

def reinitialize(system, fitparams):
    """
    Iterate a fit starting from different initial positions.
        
    The initial positions are drawn randomly from the prior, posterior or just
    kept as the one from the system, but only for those
    parameters that are adjustable and have a prior.
    
    @param system: the system to fit
    @type system: Body
    @param fitparams: fit parameters
    @type fitparams: ParameterSet
    """
    init_from = fitparams.get('init_from', 'system')
    if init_from in ['prior', 'posterior']:
        # Walk over the system and set the value of all adjustable parameters
        # that have a prior to a value randomly dranw from that prior
        for parset in system.walk():
            #-- for each parameterSet, walk through all the parameters
            for qual in parset:
                #-- extract those which need to be fitted
                is_adjust = parset.get_adjust(qual)
                has_prior = parset.has_prior(qual)
                has_post = parset.has_posterior(qual)
                
                if is_adjust and has_prior and init_from == 'prior':
                    parset.get_parameter(qual).set_value_from_prior()
                elif is_adjust and has_prior and has_post and init_from == 'posterior':
                    parset.get_parameter(qual).set_value_from_posterior()
                elif is_adjust and has_prior and not has_post:
                    logger.warning("Cannot re-initialize parameter {} from posterior (it has none)".format(qual))
        logger.warning("Re-initialized values of adjustable parameters from {}".format(init_from))
        




def monte_carlo(system, fitparams):
    """
    Add Monte Carlo noise to the datasets.
    
    We keep a copy of the "original" data also in the dataset, otherwise we're
    increasing noise all the time. Each time we add noise, we need to add it to
    the **original** dataset.
    
    @param system: the system to fit
    @type system: Body
    @param fitparams: fit parameters
    @type fitparams: ParameterSet
    """
    do_monte_carlo = fitparams.get('monte_carlo', False)
    if do_monte_carlo:
        # Walk through all the observational datasets.
        for parset in system.walk():
            # If it's an observational dataset, enter it and look for sigma
            # columns.
            if parset.get_context()[-3:] == 'obs':
                sigma_cols = [c for c in parset['columns'] if 'sigma' in c]
                for sigma in sigma_cols:
                    meas_col = sigma.split('sigma_')[1]
                    original_meas_col = '_o_' + measurement_col
                    if not original_meas_col in parset['columns']:
                        # add original-data column to data
                        parset['columns'] = parset['columns'] + [original_meas_col]
                        parset.add_parameter(qualifier=original_meas_col, 
                                             value=parset[meas_col])
                    # Add noise to data
                    noise = np.random.normal(sigma=parset[sigma])
                    parset[meas_col] = parset[original_meas_col] + noise
        logger.warning("Added simulated noise for Monte Carlo simulations")


def subsets_via_flags(system, fitparams):
    """
    Take a subset from the observations via extraction of unique flags.
    """
    do_subset = fitparams.get('subsets_via_flags', False)
    if do_subset:
        # Walk through all the observational datasets.
        for parset in system.walk():
            # If it's an observational dataset, enter it and look for flag
            # columns.
            if parset.get_context()[-3:] == 'obs':
                # Keep track of original DataSet
                original_parset = parset.copy()
                # Look for flag columns
                flag_cols = [c for c in parset['columns'] if c.split('_')[0] == 'flag']
                for flag in flag_cols:
                    unique_flags = np.sort(np.unique(self[flag]))
                    for unique_flag in unique_flags:
                        parset.take(self[flag] == unique_flag)
                        yield unique_flag,parset['ref']
                        # Restore the DataSet to its original contents
                        parset.overwrite_value_from(original_parset)
    else:
        yield None,None


#}





#{ MCMC samplers.
    
def run_pymc(system,params=None,mpi=None,fitparams=None):
    """
    Perform MCMC sampling of the parameter space of a system using pymc.
    
    Be sure to set the parameters to fit to be adjustable in the system.
    Be sure to set the priors of the parameters to fit in the system.
        
    @param system: the system to fit
    @type system: Body
    @param params: computation parameters
    @type params: ParameterSet
    @param mpi: mpi parameters
    @type mpi: ParameterSet
    @param fitparams: fit algorithm parameters
    @type fitparams: ParameterSet
    @return: the MCMC sampling history (ParameterSet of context 'fitting:pymc'
    @rtype: ParameterSet
    """
    if fitparams is None:
        fitparams = parameters.ParameterSet(frame='phoebe',context='fitting:pymc')
    # We need unique names for the parameters that need to be fitted, we need
    # initial values and identifiers to distinguish parameters with the same
    # name (we'll also use the identifier in the parameter name to make sure
    # they are unique). While we are iterating over all the parameterSets,
    # we'll also have a look at what context they are in. From this, we decide
    # which fitting algorithm to use.
    ids = []
    pars = {}
    
    #-- walk through all the parameterSets available. This needs to be via
    #   this utility function because we need to iteratively walk down through
    #   all BodyBags too.
    walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
    frames = []
    for parset in walk:
        frames.append(parset.frame)
        #-- for each parameterSet, walk through all the parameters
        for qual in parset:
    
            #-- extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                
                #-- ask a unique ID and check if this parameter has already
                #   been treated. If so, continue to the next one.
                parameter = parset.get_parameter(qual)
                myid = parameter.get_unique_label()
                if myid in ids:
                    continue
                
                #-- else, add the name to the list of pnames. Ask for the
                #   prior of this object
                name = '{}_{}'.format(qual, myid)
                pars[name] = parameter.get_prior(name=name, fitter='pymc')
                logger.warning('Fitting {} with prior {}'.format(qual,parameter.get_prior(name=name,fitter=None)))
                
                #-- and add the id
                ids.append(myid)
    
     #-- derive which algorithm to use for fitting. If all the contexts are the
    #   same, it's easy. Otherwise, it's ambiguous and we raise a ValueError
    algorithm = set(frames)
    if len(algorithm)>1:
        raise ValueError("Ambiguous set of parameters (different frames, found): {}".format(algorithm))
    else:
        algorithm = list(algorithm)[0]
        logger.info('Choosing back-end {}'.format(algorithm))
    
    
    mu,sigma,model = system.get_model()
    #mu,sigma,model = system.get_data()
    n_data = len(mu)
    
    
    def model_eval(*args,**kwargs):
        #-- evaluate the system, get the results and return a probability
        had = []
        #-- walk through all the parameterSets available:
        walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
        for parset in walk:
            #-- for each parameterSet, walk to all the parameters
            for qual in parset:
                #-- extract those which need to be fitted
                if parset.get_adjust(qual) and parset.has_prior(qual):
                    #-- ask a unique ID and update the value of the parameter
                    myid = parset.get_parameter(qual).get_unique_label()
                    if myid in had: continue
                    parset[qual] = kwargs['{}_{}'.format(qual,myid)]
                    had.append(myid)
        system.reset()
        system.clear_synthetic()
        try:
            system.compute(params=params,mpi=mpi)
            mu,sigma,model = system.get_model()
        except:
            model = np.zeros(n_data)
        return model
   
    
    #-- define the model
    mymodel = pymc.Deterministic(eval=model_eval,name='model',parents=pars,doc='Once upon a time there were three bears',\
                trace=True,verbose=1,dtype=np.array,plot=False,cache_depth=2)
    #-- define the observations
    observed = pymc.Normal('alldata',mu=mymodel,tau=1./sigma**2,value=mu,observed=True)
    pars['observed'] = observed
    #-- run the MCMC
    dbname = fitparams['label']+'.pymc.pickle'
    if fitparams['incremental'] is True and os.path.isfile(dbname):
        db = pymc.database.pickle.load(dbname)
        logger.warning('Starting from previous results {}'.format(fitparams['label']))
    else:
        db = 'pickle'
        logger.warning("Starting new chain {}".format(fitparams['label']))
    
    mc = pymc.MCMC(pars,db=db,dbname=dbname)
    #-- store the info in a feedback dictionary
    feedback = dict(parset=fitparams,parameters=[],traces=[],priors=[])
    
    mc.sample(iter=fitparams['iters'],burn=fitparams['burn'],thin=fitparams['thin'])
    if fitparams['incremental']:
        mc.db.close()
    #-- add the posteriors to the parameters
    had = []
    #-- walk through all the parameterSets available:
    walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
    for parset in walk:
        #-- fore ach parameterSet, walk to all the parameters
        for qual in parset:
            #-- extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                #-- ask a unique ID and add the parameter
                myid = parset.get_parameter(qual).get_unique_label()
                if myid in had: continue
                had.append(myid)
                #-- add the trace, priors and the parameters themselves to
                #   the dictionary
                this_param = parset.get_parameter(qual)
                #-- access all samples, also from previous runs
                trace = mc.trace('{}_{}'.format(qual,myid),chain=None)[:]
                feedback['parameters'].append(this_param)
                feedback['traces'].append(trace)
                feedback['priors'].append(this_param.get_prior(fitter=None))
    fitparams['feedback'] = feedback
    return fitparams




def run_emcee(system, params=None, fitparams=None, mpi=None, usercosts=None):
    """
    Perform MCMC sampling of the parameter space of a system using emcee.
    
    Be sure to set the parameters to fit to be adjustable in the system.
    Be sure to set the priors of the parameters to fit in the system.
    
    **Tips**
    
    * Have a look at the C{acortime}, it is good practice to let the sampler
      run at least 10 times the C{acortime}. If ``acortime = np.nan``, you
      probably need to take more iterations!
    * Use as many walkers as possible (hundreds for a handful of
      parameters)
    * Beware of a burn-in period. The most conservative you can get is making
      a histogram of only the final states of all walkers.
    
    Reference: [Foreman-Mackey2012]_
    
    .. note::
    
        Thanks to G. Matijevic.
    
    @param system: the system to fit
    @type system: Body
    @param params: computation parameters
    @type params: ParameterSet
    @param mpi: mpi parameters
    @type mpi: ParameterSet
    @param fitparams: fit algorithm parameters
    @type fitparams: ParameterSet
    @return: the MCMC sampling history (ParameterSet of context 'fitting:emcee'
    @rtype: ParameterSet
    """
    # Pickle args and kwargs in NamedTemporaryFiles, which we will delete
    # afterwards
    if not mpi or not 'directory' in mpi or not mpi['directory']:
        direc = os.getcwd()
    else:
        direc = mpi['directory']
        
    # Check if nonadjustables with priors are inside their limits:
    for par in system.get_parameters_with_priors(is_adjust=False):
        if np.isinf(par.get_logp()):
            raise ValueError(("At least one parameter that is not adjustable has "
                              "an impossible value according to its prior or limits: "
                              "qualifier={}, prior={}, limits={}".format(par.get_qualifier(),
                              par.get_prior(), par.get_limits())))
            
    
    # The system
    sys_file = tempfile.NamedTemporaryFile(delete=False, dir=direc)
    pickle.dump(system, sys_file)
    sys_file.close()
    
    # The compute params
    compute_file = tempfile.NamedTemporaryFile(delete=False, dir=direc)
    pickle.dump(params, compute_file)
    compute_file.close()
    
    # The fit params
    fit_file = tempfile.NamedTemporaryFile(delete=False, dir=direc)
    pickle.dump(fitparams, fit_file)
    fit_file.close()
    
    # The user costs
    if usercosts is None:
        usercosts = UserCosts([])
    usercosts_file = tempfile.NamedTemporaryFile(delete=False, dir=direc)
    pickle.dump(usercosts, usercosts_file)
    usercosts_file.close()
    
    # Create arguments to run emceerun_backend.py
    args = " ".join([sys_file.name, compute_file.name, fit_file.name, usercosts_file.name, fit_logger_level])
    
    # Then run emceerun_backend.py
    if mpi is not None:
        emceerun_backend.mpi = True
        flag, mpitype = decorators.construct_mpirun_command(script='emceerun_backend.py',
                                                  mpirun_par=mpi, args=args)

        #flag = subprocess.call(cmd, shell=True)
                
        # If something went wrong, we can exit nicely here, the traceback
        # should be printed at the end of the MPI process
        if flag:
            sys.exit(1)
    
    else:
        emceerun_backend.mpi = False
        emceerun_backend.run(sys_file.name, compute_file.name, fit_file.name, usercosts_file.name)
            
    # Clean up pickle files once they are loaded:
    os.unlink(sys_file.name)
    os.unlink(fit_file.name)
    os.unlink(compute_file.name)
    
    # Check if we produced the chain file
    chain_file = os.path.join(direc, fitparams['label'] + '.mcmc_chain.dat')
    
    if not os.path.isfile(chain_file):
        raise RuntimeError("Could not produce chain file {}, something must have seriously gone wrong during emcee run".format(chain_file))
        
    return chain_file, 
    
    
#}

#{ Nonlinear optimizers


def run_lmfit(system, params=None, mpi=None, fitparams=None):
    """
    Perform nonlinear fitting of a system using lmfit.
    
    Be sure to set the parameters to fit to be adjustable in the system.
    
    @param system: the system to fit
    @type system: Body
    @param params: computation parameters
    @type params: ParameterSet
    @param mpi: mpi parameters
    @type mpi: ParameterSet
    @param fitparams: fit algorithm parameters
    @type fitparams: ParameterSet
    @return: the MCMC sampling history (ParameterSet of context 'fitting:pymc'
    @rtype: ParameterSet
    """
    
    # We need some information on how to fit exactly; if the user didn't give
    # it we'll use the defaults
    if fitparams is None:
        fitparams = parameters.ParameterSet(frame='phoebe',
                                            context='fitting:lmfit')
    
    # For lmfit, the initial values need to make sense
    # Check if everything is OK (parameters are inside limits and/or priors)
    passed, errors = system.check(return_errors=True)
    if not passed:
        raise ValueError(("Some parameters are outside of reasonable limits or "
                          "prior bounds: {}").format(", ".join(errors)))
    
    # We need unique names for the parameters that need to be fitted, we need
    # initial values and identifiers to distinguish parameters with the same
    # name (we'll also use the identifier in the parameter name to make sure
    # they are unique). While we are iterating over all the parameterSets,
    # we'll also have a look at what context they are in. From this, we decide
    # which fitting algorithm to use.
    ids = []
    pars = lmfit.Parameters()
    ppars = [] # Phoebe parameters
    init_ppars = []
    
    #-- walk through all the parameterSets available. This needs to be via
    #   this utility function because we need to iteratively walk down through
    #   all BodyBags too.
    frames = []
    for parset in system.walk():
        frames.append(parset.frame)
        
        # If the parameterSet is not enabled, skip it
        if not parset.get_enabled():
            continue
        
        #-- for each parameterSet, walk through all the parameters
        for qual in parset:
            
            #-- extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                
                #-- ask a unique ID and check if this parameter has already
                #   been treated. If so, continue to the next one.
                parameter = parset.get_parameter(qual)
                myid = parameter.get_unique_label()
                if myid in ids:
                    continue
                
                #-- else, add the name to the list of pnames. Ask for the
                #   prior of this object
                name = '{}_{}'.format(qual, myid).replace('-','_')
                prior = parameter.get_prior()
                if fitparams['bounded']:
                    minimum, maximum = prior.get_limits(factor=fitparams['bounded'])
                else:
                    minimum, maximum = None, None
                pars.add(name, value=parameter.get_value(),
                         min=minimum, max=maximum, vary=True)
                
                #-- and add the id
                ids.append(myid)
                ppars.append(parameter)
                init_ppars.append(parameter.copy())
    
    # Current state: we know all the parameters that need to be fitted (and
    # their values), and we have unique string labels to refer to them.
    
    # Next up: define a function to evaluate the model given a set of
    # parameter values. You'll see that a lot of the code is similar as the
    # one above; but instead of just keeping track of the parameters, we
    # change the value of each parameter. This function will be called by the
    # fitting algorithm itself.
    
    traces = []
    redchis = []
    Nmodel = dict()
    
    def model_eval(pars, system):
        # Evaluate the system, get the results and return a probability
        
        # Remember the parameters we already set
        had = []
        
        # walk through all the parameterSets available:
        for parset in system.walk():
            
            # If the parameterSet is not enabled, skip it
            if not parset.get_enabled():
                continue
            
            # for each parameterSet, walk to all the parameters
            for qual in parset:
                
                # extract those which need to be fitted
                if parset.get_adjust(qual) and parset.has_prior(qual):
                    
                    # ask a unique ID and update the value of the parameter
                    myid = parset.get_parameter(qual).get_unique_label().replace('-', '_')
                    if myid in had:
                        continue
                    parset[qual] = pars['{}_{}'.format(qual, myid)].value
                    had.append(myid)
                    
        # Current state: the system has been updated with the newly proposed
        # values.
        
        # Next up: compute the model given the current (new) parameter values
        system.reset()
        system.clear_synthetic()
        system.compute(params=params, mpi=mpi)
        
        # The fitting algorithm needs the model as one long array of data.
        # The 'get_model' does exactly that: even if you have 10 light curves,
        # they will all be concatenated. It does the same with the observations
        # themself (mu, sigma)
        mu, sigma, model = system.get_model()
        retvalue = (model - mu) / sigma
        
        # short log message to report to the user:
        names = [par.get_qualifier() for par in ppars]
        vals = [pars[par].value for par in pars]
        logger.warning("Current values: {} (chi2={:.6g})".format(", ".join(['{}={}'.format(name,val) for name,val in zip(names,vals)]),(retvalue**2).mean()))
        
        # keep track of trace value and chi2 value for future reference and
        # introspection
        traces.append(vals)
        redchis.append(np.array(retvalue**2).sum() / (len(model)-len(pars)))
        Nmodel['Nd'] = len(model)
        Nmodel['Np'] = len(pars)

        # That's it concerning model evaluation!
        return retvalue
    
    # Current state: we know which parameters to fit, and we have a function
    # that returns a fit statistic given a proposed set of parameter values.
    
    # Next up: run the fitting algorithm!
    
    # The user can give fine tuning parameters if the fitting parameterSet is a
    # subcontext of fitting:lmfit
    context_hierarchy = fitparams.get_context().split(':')
        
    #-- do the fit and report on the errors to the screen
    
    if len(context_hierarchy) == 2:
        
        #   we suffer quite drastically from numerical effects, i.e. that for
        #   example the total integrated flux changes relatively drastically for
        #   very small inclination changes. This messes up the leastsq algorithm,
        #   so we need to tell it that it has to take wide steps for the computation
        #   of its derivatives.
    
        extra_kwargs = {}
        if fitparams['method'] == 'leastsq':
            extra_kwargs['epsfcn'] = 1e-3
        
        # Then we can safely run the fitter.
        try:
            result = lmfit.minimize(model_eval, pars, args=(system,),
                            method=fitparams['method'], **extra_kwargs)
        except Exception as msg:
            raise RuntimeError(("Error in running lmfit. Perhaps the version "
                "is not right? (you have {} and should have >{}). Original "
                "error message: {}").format(lmfit.__version__, '0.7', str(msg)))
    
    # In this case we have fine tuning parameters
    else:
        # Few things to do:
        # 1. Construct a general "Minimizer" object
        # 2. Extract extra fit fine tuning parameters from ParameterSet
        # 3. Run the algorithm
        method = context_hierarchy[-1].lower()
        result = lmfit.Minimizer(model_eval, pars, fcn_args=(system,),
                                    scale_covar=True)
        
        _methods = {'anneal': 'anneal',
               'nelder': 'fmin',
               'lbfgsb': 'lbfgsb',
               'leastsq': 'leastsq'}
        
        _scalar_methods = {'nelder': 'Nelder-Mead',
                       'powell': 'Powell',
                       'cg': 'CG ',
                       'bfgs': 'BFGS',
                       'newton': 'Newton-CG',
                       'anneal': 'Anneal',
                       'lbfgs': 'L-BFGS-B',
                       'l-bfgs': 'L-BFGS-B',
                       'tnc': 'TNC',
                       'cobyla': 'COBYLA',
                       'slsqp': 'SLSQP'}
        
        # Specify fine-tuning parameters
        _allowed_kwargs = {'nelder': ['ftol', 'xtol', 'maxfun'],
                           'leastsq': ['ftol', 'xtol', 'gtol', 'maxfev','epsfcn']}
        
        fitoptions = dict()
        for key in _allowed_kwargs[method]:
            # only override when value is not None
            if fitparams[key] is not None:
                fitoptions[key] = fitparams[key]
        if method in _methods:
            getattr(result, _methods[method])(**fitoptions)
        elif method in _scalar_methods:
            getattr(result, _scalar_methods[method])(**fitoptions)
        else:
            raise ValueError("Unknown method '{}'".format(method))        
    
    # Current state: we found the best set of parameters! The system is fitted!
    
    # Next up: some bookkeeping and reporting
    
    lmfit.report_errors(pars)

    # Extract the values to put them in the feedback
    if hasattr(result, 'success') and not result.success:
        logger.error("nonlinear fit with method {} failed".format(fitparams['method']))
        values = [np.nan for ipar in pars]
        success = result.success
        redchi = result.redchi
    else:
        values = [pars['{}_{}'.format(ipar.get_qualifier(), ipar.get_unique_label().replace('-','_'))].value for ipar in ppars]
        success = None
        redchi = None
        
    #-- the same with the errors and correlation coefficients, if there are any
    if hasattr(result, 'errorbars') and result.errorbars:
        sigmas = [pars['{}_{}'.format(ipar.get_qualifier(), ipar.get_unique_label().replace('-','_'))].stderr for ipar in ppars]
        correl = [pars['{}_{}'.format(ipar.get_qualifier(), ipar.get_unique_label().replace('-','_'))].correl for ipar in ppars]
    else:
        sigmas = [np.nan for ipar in pars]
        correl = [np.nan for ipar in pars]
        logger.error("Could not estimate errors (set to nan)")
    
    #-- when asked, compute detailed confidence intervals
    if fitparams['compute_ci']:
        #-- if we do this, we need to disable boundaries
        for name in pars:
            pars[name].min = None
            pars[name].max = None
        try:
            ci = lmfit.conf_interval(result, sigmas=(0.674, 0.997),
                                     verbose=True, maxiter=10)
            lmfit.printfuncs.report_ci(ci)
        except Exception as msg:
            logger.error("Could not estimate CI (original error: {}".format(str(msg)))
    
    traces = np.array(traces).T
    
    extra_info = dict(traces=traces, redchis=redchis, Ndata=Nmodel['Nd'], Npars=Nmodel['Np'])
    
    return result, pars, extra_info


class MinuitMinimizer(object):
    def __init__(self,system,params=None,mpi=None):
        self.system = system
        self.params = params # computational parameters
        self.mpi = mpi
        
        #-- evaluate the system, get the results and return a probability
        had = []
        #-- walk through all the parameterSets available:
        #walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
        self.param_names = []
        for parset in system.walk():
            #-- for each parameterSet, walk to all the parameters
            for qual in parset:
                #-- extract those which need to be fitted
                if parset.get_adjust(qual) and parset.has_prior(qual):
                    #-- ask a unique ID and update the value of the parameter
                    this_param = parset.get_parameter(qual)
                    myid = this_param.get_unique_label().replace('-','_')
                    if myid in had: continue
                    id_for_minuit = '{}_{}'.format(qual,myid)
                    setattr(self,id_for_minuit,this_param.get_value())
                    had.append(myid)
                    self.param_names.append(id_for_minuit)
                    
        self.func_code = iminuit.util.make_func_code(self.param_names)
        self.func_defaults = None
        #-- construct the iminuit Minimizer
    
    def __call__(self,*pars):
        """
        Define the chi square.
        """
        system = self.system
        had = []
        #-- walk through all the parameterSets available:
        #walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
        for parset in system.walk():
            #-- for each parameterSet, walk to all the parameters
            for qual in parset:
                #-- extract those which need to be fitted
                if parset.get_adjust(qual) and parset.has_prior(qual):
                    #-- ask a unique ID and update the value of the parameter
                    myid = parset.get_parameter(qual).get_unique_label().replace('-','_')
                    if myid in had: continue
                    id_for_minuit = '{}_{}'.format(qual,myid)
                    index = self.param_names.index(id_for_minuit)
                    parset[qual] = pars[index]
                    had.append(myid)
        #-- evaluate the system, get the results and return a probability
        system.reset()
        system.clear_synthetic()
        system.compute(params=self.params,mpi=self.mpi)
        mu,sigma,model = system.get_model()
        retvalue = (model-mu)/sigma
        
        print(", ".join(["{}={}".format(name,pars[i]) for i,name in enumerate(self.param_names)]))
        print("Current value",sum(retvalue**2))
        return sum(retvalue**2)


def run_minuit(system,params=None,mpi=None,fitparams=None):
    """
    Use MINUIT.
    
    MINUIT2, is a numerical minimization computer program originally written
    in the FORTRAN programming language[1] by CERN staff physicist Fred James
    in the 1970s (from wikipedia).
    
    Best to call Minos to check if the parameters are reasonably independent.
    The tool is MnHesse.
    
    .. note::
        
        Thanks to B. Leroy for detailed explanations. He is not responsible
        for any bugs or errors.
        
    """
    mimi = MinuitMinimizer(system)
    had = []
    ppars = [] # Phoebe parameters
    pars = {}  # Minuit parameters
    #-- walk through all the parameterSets available:
    #walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
    for parset in system.walk():
        #-- for each parameterSet, walk to all the parameters
        for qual in parset:
            #-- extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                #-- ask a unique ID and update the value of the parameter
                this_param = parset.get_parameter(qual)
                myid = this_param.get_unique_label().replace('-','_')
                if myid in had: continue
                id_for_minuit = '{}_{}'.format(qual,myid)
                pars[id_for_minuit] = this_param.get_value()
                if this_param.has_step():
                    pars['error_{}'.format(id_for_minuit)] = this_param.get_step()
                else:
                    #-- this is MINUIT's default, we'll set it otherwise we get a warning
                    pars['error_{}'.format(id_for_minuit)] = 1.
                if fitparams['bounded']:
                    lower,upper = this_param.get_prior().get_limits(factor=fitparams['bounded'])
                    pars['limit_{}'.format(id_for_minuit)] = (lower,upper)
                had.append(myid)
                ppars.append(this_param)
    if fitparams['bounded']:
        logger.warning("MINUIT: parameters are bounded, this can affect the error estimations")
    #-- a change of this value should be interpreted as 1 sigma for iminuit:
    # (in order to calculate it we need to now the number of observations)
    data,sigma = system.get_data()
    N = len(data)-len(ppars)
    errordef = np.sqrt(2.0*(N-2.0))
    logger.warning("MINUIT's 1sigma = {}".format(errordef))
    #-- initialize the fit
    m = iminuit.Minuit(mimi,errordef=errordef,**pars)
    #-- run the fit
    m.migrad()
    #-- collect the feedback:
    print(dir(m))
    values = [m.values[x] for x in m.parameters]
    sigmas = [m.errors[x] for x in m.parameters]
    #-- note: m.covariance is also available, as well as m.matrix(correlation=True)
    #   there is also stuff for contours etc... google for it! But we probably
    #   want to add the info to the feedback
    feedback = dict(parameters=ppars,values=values,sigmas=sigmas,redchi=m.fval,
                    message=m.get_fmin(),success=m.get_fmin().is_valid)
    fitparams['feedback'] = feedback
    return fitparams



#}

#{ Gridding


def run_grid(system,params=None,mpi=None,fitparams=None):
    """
    Compute a grid in the adjustable parameter space.
    
    Be sure to set the parameters to grid to be adjustable in the system.
    Be sure to set the priors to have the ``discrete`` distribution, with the
    ``values`` the points in the grid. If you want a real grid, you need to set
    the ``iterate`` parameter in the fitting context to ``product``. If you just
    want to compute the system in a set of parameter combinations, set it to
    ``list``. In the latter case, all ``values`` from the priors need to have
    the same length and that is equal to the number of models that will be
    computed. In the ``product`` case, the number of models equals the product
    of all lengths of ``values``. This can explode rapidly!
    
    **Examples setup:**
    
    Specify the parameters to iterate over, as well as their priors:
    
    >>> star.set_adjust(('mass','radius'),True)
    >>> star.get_parameter('mass').set_prior(distribution='discrete',values=[1.0,1.1,1.3])
    >>> star.get_parameter('radius').set_prior(distribution='discrete',values=[0.7,0.8,1.0])
    
    Specify the gridding parameters:
    
    >>> fitparams = phoebe.ParameterSet('fitting:grid',iterate='product')
    
    And run the fit:
    
    >>> feedback = run(system,fitparams=fitparams)
    
    **See also:**
    
    - :ref:`fitting:grid <parlabel-phoebe-fitting:grid>` 
    - :ref:`feedback <label-feedback-fitting:grid-phoebe>`, :ref:`iterate <label-iterate-fitting:grid-phoebe>`
    
    @param system: the system to fit
    @type system: Body
    @param params: computation parameters
    @type params: ParameterSet
    @param mpi: mpi parameters
    @type mpi: ParameterSet
    @param fitparams: fit algorithm parameters
    @type fitparams: ParameterSet
    @return: the grid sampling history (ParameterSet of context 'fitting:grid'
    @rtype: ParameterSet
    """
    if fitparams is None:
        fitparams = parameters.ParameterSet(frame='phoebe',context='fitting:grid')
    
    sampling = fitparams['sampling']
    
    # We need unique names for the parameters that need to be fitted, we need
    # initial values and identifiers to distinguish parameters with the same
    # name (we'll also use the identifier in the parameter name to make sure
    # they are unique). While we are iterating over all the parameterSets,
    # we'll also have a look at what context they are in. From this, we decide
    # which fitting algorithm to use.
    ids = []
    names = []
    ranges = []
    
    # Walk through all the parameterSets available. This needs to be via this
    # this utility function because we need to iteratively walk down through all
    # BodyBags too.
    for parset in system.walk():
        
        # For each parameterSet, walk through all the parameters
        for qual in parset:
            
            # Extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                
                # Ask a unique ID and check if this parameter has already been
                # treated. If so, continue to the next one.
                parameter = parset.get_parameter(qual)
                myid = parameter.get_unique_label()
                if myid in ids:
                    continue
                
                # Construct the range:
                myrange = parameter.get_prior().get_grid(sampling)
                
                # And add the id
                ids.append(myid)
                names.append(qual)
                ranges.append(myrange) 
    
    def lnprob(pars, ids, system):
        
        # Evaluate the system, get the results and return a probability
        had = []
        
        # Walk through all the parameterSets available:
        #walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
        for parset in system.walk():
            #-- for each parameterSet, walk to all the parameters
            for qual in parset:
                #-- extract those which need to be fitted
                if parset.get_adjust(qual) and parset.has_prior(qual):
                    #-- ask a unique ID and update the value of the parameter
                    myid = parset.get_parameter(qual).get_unique_label()
                    if myid in had: continue
                    index = ids.index(myid)
                    parset[qual] = pars[index]
                    had.append(myid)
        system.reset()
        system.clear_synthetic()
        system.compute(params=params,mpi=mpi)
        
        logp, chi2, N = system.get_logp()
        return logp
    
    #-- now run over the whole grid
    grid_pars = []
    grid_logp = []
    save_files= []
    
    #-- well, that is, it can be defined as a product or a list:
    #   product of [1,2], [10,11] is [1,10],[1,11],[2,10],[2,11]
    #   list of    [1,2], [10,11] is [1,10],[2,11]
    if fitparams['iterate'] == 'product':
        the_iterator = itertools.product(*ranges)
    elif fitparams['iterate'] == 'list':
        the_iterator = zip(*ranges)
    
    #-- now run over the whole grid
    for i,pars in enumerate(the_iterator):
        msg = ', '.join(['{}={}'.format(j,k) for j,k in zip(names,pars)])
        mylogp = lnprob(pars,ids,system)
        logger.warning('GRID: step {} - parameters: {} (logp={:.3g})'.format(i,msg,mylogp))
        this_system = copy.deepcopy(system)
        this_system.remove_mesh()
        this_system.save('gridding_{:05d}.phoebe'.format(i))
        del this_system
        grid_pars.append(pars)
        grid_logp.append(mylogp)
        save_files.append('gridding_{:05d}.phoebe'.format(i))
    
    #-- convert to arrays
    grid_pars = np.array(grid_pars)
    grid_logp = np.array(grid_logp)
    
    #-- store the info in a feedback dictionary
    feedback = dict(parameters=[],values=[],priors=[],save_files=[])
    
    #-- add the posteriors to the parameters
    had = []
    #-- walk through all the parameterSets available:
    #walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
    for parset in system.walk():
        #-- fore ach parameterSet, walk to all the parameters
        for qual in parset:
            #-- extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                #-- ask a unique ID and update the value of the parameter
                myid = parset.get_parameter(qual).get_unique_label()
                if myid in had: continue
                had.append(myid)
                index = ids.index(myid)
                this_param = parset.get_parameter(qual)
                #this_param.set_posterior(trace.ravel())f
                feedback['parameters'].append(this_param)
                feedback['values'].append(grid_pars[:,index])
                feedback['priors'].append(this_param.get_prior().get_grid(sampling))
    feedback['logp'] = grid_logp
    feedback['save_files'] = save_files
    fitparams['feedback'] = feedback
    return fitparams

#}


#{ Genetic algorithm


def run_genetic(system, params=None, mpi=None, fitparams=None):
    """
    Fit the system using a genetic algorithm.
        
    **See also:**
        
    @param system: the system to fit
    @type system: Body
    @param params: computation parameters
    @type params: ParameterSet
    @param mpi: mpi parameters
    @type mpi: ParameterSet
    @param fitparams: fit algorithm parameters
    @type fitparams: ParameterSet
    @return: the grid sampling history (ParameterSet of context 'fitting:genetic'
    @rtype: ParameterSet
    """
    if fitparams is None:
        fitparams = parameters.ParameterSet(frame='phoebe',
                                            context='fitting:genetic')
    
    # Get the size of the population
    population_size = fitparams['population_size']
    
    # We need unique names for the parameters that need to be fitted, we need
    # initial values and identifiers to distinguish parameters with the same
    # name (we'll also use the identifier in the parameter name to make sure
    # they are unique). While we are iterating over all the parameterSets,
    # we'll also have a look at what context they are in. From this, we decide
    # which fitting algorithm to use.
    ids = []
    names = []
    ranges = []
    
    # Walk through all the parameterSets available. This needs to be via this
    # utility function because we need to iteratively walk down through all
    # BodyBags too.
    for parset in system.walk():
        
        # For each parameterSet, walk through all the parameters
        for qual in parset:
            
            # Extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                
                # Ask a unique ID and check if this parameter has already been
                # treated. If so, continue to the next one.
                parameter = parset.get_parameter(qual)
                myid = parameter.get_unique_label()
                if myid in ids: continue
                
                # Get the limits of the parameter (or the set of initial children?)
                population = parameter.get_prior().draw(size=population_size)
                
                # And add the id
                ids.append(myid)
                names.append(qual)
                ranges.append(population) 
    
    
    raise NotImplementedError
    def lnprob(pars,ids,system):
        #-- evaluate the system, get the results and return a probability
        had = []
        #-- walk through all the parameterSets available:
        walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
        for parset in system.walk():
            #-- for each parameterSet, walk to all the parameters
            for qual in parset:
                #-- extract those which need to be fitted
                if parset.get_adjust(qual) and parset.has_prior(qual):
                    #-- ask a unique ID and update the value of the parameter
                    myid = parset.get_parameter(qual).get_unique_label()
                    if myid in had: continue
                    index = ids.index(myid)
                    parset[qual] = pars[index]
                    had.append(myid)
        system.reset()
        system.clear_synthetic()
        system.compute(params=params,mpi=mpi)
        logp, chi2, N = system.get_logp()
        return logp
    
    #-- now run over the whole grid
    grid_pars = []
    grid_logp = []
    save_files= []
    
    #-- well, that is, it can be defined as a product or a list:
    #   product of [1,2], [10,11] is [1,10],[1,11],[2,10],[2,11]
    #   list of    [1,2], [10,11] is [1,10],[2,11]
    if fitparams['iterate']=='product':
        the_iterator = itertools.product(*ranges)
    elif fitparams['iterate']=='list':
        the_iterator = zip(*ranges)
    
    #-- now run over the whole grid
    for i,pars in enumerate(the_iterator):
        msg = ', '.join(['{}={}'.format(j,k) for j,k in zip(names,pars)])
        mylogp = lnprob(pars,ids,system)
        logger.warning('GRID: step {} - parameters: {} (logp={:.3g})'.format(i,msg,mylogp))
        this_system = copy.deepcopy(system)
        this_system.remove_mesh()
        this_system.save('gridding_{:05d}.phoebe'.format(i))
        del this_system
        grid_pars.append(pars)
        grid_logp.append(mylogp)
        save_files.append('gridding_{:05d}.phoebe'.format(i))
    
    #-- convert to arrays
    grid_pars = np.array(grid_pars)
    grid_logp = np.array(grid_logp)
    
    #-- store the info in a feedback dictionary
    feedback = dict(parameters=[],values=[],priors=[],save_files=[])
    
    #-- add the posteriors to the parameters
    had = []
    #-- walk through all the parameterSets available:
    #walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
    for parset in system.walk():
        #-- fore ach parameterSet, walk to all the parameters
        for qual in parset:
            #-- extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                #-- ask a unique ID and update the value of the parameter
                myid = parset.get_parameter(qual).get_unique_label()
                if myid in had: continue
                had.append(myid)
                index = ids.index(myid)
                this_param = parset.get_parameter(qual)
                #this_param.set_posterior(trace.ravel())f
                feedback['parameters'].append(this_param)
                feedback['values'].append(grid_pars[:,index])
                feedback['priors'].append(this_param.get_prior().distr_pars['bins'])
    feedback['logp'] = grid_logp
    feedback['save_files'] = save_files
    fitparams['feedback'] = feedback
    return fitparams

#}


#{ Feedbacks

class Feedback(object):
    """
    Feedback from a fit.
    
    For inter-comparison between different fitmethods, we need to choose one
    statistic. That'll be a reduced chi-square (redchi2).
    
    These are the required properties of a feedback:
        
        - ``parameters``: the Parameter instances
        - ``values``: values of the best fit
        - ``sigmas``: estimated uncertainties or standard deviation of
          distributions
        - ``redchi``: reduced chi-square of best fit
        - ``n_data``: number of data points
        - ``n_pars``: number of free parameters
        - ``fitparams``: parameterSet used in fitting
        
        
        
    These are optional properties:
        
        - ``correls``: correlation coefficients
        - ``traces``: histories of the fit
        - ``stats``: reduced chi-squares connected to the histories.
        - ``success``: message
        
    """
    def __init__(self):
    
        self._keys = ['parameters', 'values', 'sigmas',
                      'redchi', 'n_data', 'n_pars', 'ref']
        self.index = 0
   
    def keys(self):
        return self._keys
    
    def __iter__(self):
        """
        Make the class iterable.
        """
        return self
    
    def __next__(self):
        if self.index >= len(self._keys):
            self.index = 0
            raise StopIteration
        else:
            self.index += 1
            return self._keys[self.index-1]
    
    def get_chi2(self):
        return 1.0
    
    def __getitem__(self, key):
        return getattr(self,key)
    
    def __lt__(self, other):
        """
        One statistic is less than another if it implies a worse fit.
        
        A worse fit means a larger chi square.
        """
        return self.get_chi2() > other.get_chi2()
    
    
    def __str__(self):
        """
        String representation of a feedback.
        """
        txt = "Result from fit {}\n".format(self.fitparams['ref'])
        for par, val, sig in zip(self.parameters, values, sigmas):
            qual = par.get_qualifier()
            txt+= "{:10s} = {} +/- {}\n".format(qual, val, sig)
        return txt
    
    def save(self,filename):
        ff = open(filename,'w')
        pickle.dump(self,ff)
        ff.close()  
        logger.info('Saved model to file {} (pickle)'.format(filename))
    
    next = __next__

def load(filename):
    """
    Load a class from a file.
    
    @return: Body saved in file
    @rtype: Body
    """
    ff = open(filename,'r')
    myclass = pickle.load(ff)
    ff.close()
    return myclass


#}

#{ Verifiers

def check_system(system, fitparams):
    """
    Diagnose which parameters are to be fitted, and give the user feedback when
    something is wrong.
    
    List the parameters that need to be fitted, but also the included data
    sets.
    """
    pars = []
    ids = []
    enabled = []
    logger.info('Search for all parameters to be fitted:')
    #-- walk through all the parameterSets available. This needs to be via
    #   this utility function because we need to iteratively walk down through
    #   all BodyBags too.
    for parset in system.walk():
        
        #-- for each parameterSet, walk through all the parameters
        for qual in parset:
            
            #-- extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                #-- ask a unique ID and check if this parameter has already
                #   been treated. If so, continue to the next one.
                parameter = parset.get_parameter(qual)
                myid = parameter.get_unique_label()
                if myid in ids:
                    continue
                ids.append(myid)
                #logger.info(('Parameter {} with prior {} (unique label: {}) '
                #            '').format(qual, parameter.get_prior(), myid))
                logger.info(('Parameter {:10s} with prior {}'
                            '').format(qual, parameter.get_prior()))
        
        # If this parameterSet is a DataSet, check if it is enabled
        if parset.get_context()[-3:] == 'obs' and parset.get_enabled():
            enabled.append('{} ({})'.format(parset['ref'], parset.get_context()))
    
    logger.info("The following datasets are included in the fit: {}".format(", ".join(enabled)))
            
        
    # There has to be at least one parameter to fit, otherwise it's quite silly
    # to call the fitting routine
    if not ids:
        raise ValueError(('Did not find any parameters with priors to fit. '
                          'Call "set_adjust" in the relevant ParameterSets, '
                          'and define priors for all parameters to be '
                          'fitted.'))
                           
            

def summarize_fit(system,fitparams,correlations=None,savefig=False):
    """
    Summarize a fit.
    
    @param system: the fitted system
    @type system: Body
    @param fitparams: the fit parameters (containing feedback)
    @type fitparams: ParameterSet
    """
    #-- it might be useful to know what context we are plotting
    context = fitparams.get_context().split(':')[1]
    #-- correlations can be plotted as hexbin histograms or as walks. For
    #   MCMC fitters it makes more sense to use hexbin as a default, and for
    #   LMFIT it makes more sense to use a walk.
    if correlations is None:
        if context=='lmfit':
            correlations = 'walk'
        else:
            correlations = 'hexbin'
    #-- which parameters were fitted?
    fitted_param_labels = [par.get_unique_label() for par in fitparams['feedback']['parameters']]
    #-- remember that we could also have defined constraints to derive other
    #   parameters. We need to build traces for these so that we can report
    #   on them.
    constrained_parameters = dict()
    
    #-- walk through all parameterSets until you encounter a parameter that
    #   was used in the fit
    had = []
    indices = []
    paths = []
    units = []
    table = [] # table from trace
    table_from_fitter = [] # table from fitter
    for path,param in system.walk_all(path_as_string=True):
        if not isinstance(param,parameters.Parameter):
            continue
        #-- get the parameter's unique identification string
        this_label = param.get_unique_label()
        #-- now, we found a parameter that was fitted and not seen before, if
        #   the following statement is true
        if this_label in fitted_param_labels and not this_label in had:
            if 'orbit' in path:
                orb_index = path.index('orbit')
                path = path[:orb_index-1]+path[orb_index:]
            #-- where is this parameter located in the feedback?
            index = fitted_param_labels.index(this_label)
            #-- recover its trace and prior if present
            trace = fitparams['feedback']['traces'][index]
            if 'priors' in fitparams['feedback']:
                prior = fitparams['feedback']['priors'][index]
            else:
                prior = None
            # compute the Raftery-Lewis diagnostics (this only makes sense for
            # MCMC walks but anyway)
            req_iters,thin_,burn_in,further_iters,thin = pymc.raftery_lewis(trace,q=0.025,r=0.1,verbose=0)
            thinned_trace = trace[burn_in::thin]
            indtrace = np.arange(len(trace))
            indtrace = indtrace[burn_in::thin]
            if param.has_unit():
                unit = param.get_unit()
            else:
                unit=''
                
            #-- now, create the figure    
            plt.figure(figsize=(14,8))
            
            #-- first subplot contains the marginalised distribution
            plt.subplot(221)
            plt.title("Marginalised distribution")
            #-- bins according to Scott's normal reference rule
            if len(trace)>3 and np.std(trace)>0:
                bins = int(trace.ptp()/(3.5*np.std(trace)/len(trace)**0.333))
                if bins>0:
                    plt.hist(trace,bins=bins,alpha=0.5,normed=True)
            if len(thinned_trace)>3 and np.std(thinned_trace)>0:
                bins_thinned = int(thinned_trace.ptp()/(3.5*np.std(thinned_trace)/len(thinned_trace)**0.333))
                if bins_thinned>0:
                    plt.hist(thinned_trace,bins=bins_thinned,alpha=0.5,normed=True)
            #-- if a prior was defined, mark it in the plot
            if prior is not None:
                if prior.distribution=='uniform':
                    plt.axvspan(prior.distr_pars['lower'],prior.distr_pars['upper'],alpha=0.05,color='r')
                elif prior.distribution=='normal':
                    plt.axvspan(prior.distr_pars['mu']-1*prior.distr_pars['sigma'],prior.distr_pars['mu']+1*prior.distr_pars['sigma'],alpha=0.05,color='r')
                    plt.axvspan(prior.distr_pars['mu']-3*prior.distr_pars['sigma'],prior.distr_pars['mu']+3*prior.distr_pars['sigma'],alpha=0.05,color='r')
            plt.xlabel("/".join(path)+' [{}]'.format(unit))
            plt.ylabel('Number of occurrences')
            plt.grid()
            
            #-- second subplot is the Geweke plot
            plt.subplot(243)
            if context=='lmfit':
                redchis = fitparams['feedback']['redchis']
                k = max(fitparams['feedback']['Ndata'] - fitparams['feedback']['Npars'],1)
                ci = scipy.stats.distributions.chi2.cdf(redchis,k)
                plt.plot(trace,redchis,'ko')
                plt.axhline(scipy.stats.distributions.chi2.isf(0.1,k),lw=2,color='r',linestyle='-')
                plt.axhline(scipy.stats.distributions.chi2.isf(0.5,k),lw=2,color='r',linestyle='--')
                plt.gca().set_yscale('log')
                plt.xlabel("/".join(path)+' [{}]'.format(unit))
                plt.ylabel("log10 ($\chi^2$)")
                plt.grid()
            else:
                plt.title("Geweke plot")
                scores = np.array(pymc.geweke(trace)).T
                plt.plot(scores[0],scores[1],'o')
                plt.axhline(-2,color='g',linewidth=2,linestyle='--')
                plt.axhline(+2,color='b',linewidth=2,linestyle='--')
                plt.grid()
                plt.xlabel("Iteration")
                plt.ylabel("Score")
            
            #-- third subplot is the Box-and-whiskers plot
            plt.subplot(244)
            plt.title("Box and whisper plot")
            plt.boxplot(trace)
            plt.ylabel("/".join(path)+' [{}]'.format(unit))
            txtval = '{}=({:.3g}$\pm${:.3g}) {}'.format(path[-1],trace.mean(),trace.std(),unit)
            plt.xlabel(txtval)#,xy=(0.05,0.05),ha='left',va='top',xycoords='axes fraction')
            plt.xticks([])
            plt.grid()
            
            #-- and finally the trace
            plt.subplot(212)
            plt.plot(indtrace,thinned_trace,'b-',lw=2,alpha=0.5)
            plt.plot(trace,'g-')
            plt.xlim(0,len(trace))
            plt.xlabel("Iteration")
            plt.ylabel("/".join(path)+' [{}]'.format(unit))
            plt.grid()
            if savefig:
                plt.savefig("_".join(path).replace('.','_')+'.png')
            
            
            had.append(this_label)
            indices.append(index)
            paths.append(path)
            units.append(unit)
            
            table.append(["/".join(path),'{:.3g}'.format(trace.mean()),
                                         '{:.3g}'.format(trace.std()),
                                         '{:.3g}'.format(np.median(trace)),unit])
            
            if context=='lmfit':
                table_from_fitter.append(["/".join(path),
                                         '{:.3g}'.format(fitparams['feedback']['values'][index]),
                                         '{:.3g}'.format(fitparams['feedback']['sigmas'][index]),
                                         'nan',unit])
        #-- perhaps the parameter was a constraint?
        #elif not this_label in had:
            #if param.get_qualifier()=='mass':
                #print param.get_qualifier()
    
    #-- print a summary table
    #-- what is the width of the columns?
    col_widths = [9,4,3,3,4]
    for line in table:
        for c,col in enumerate(line):
            col_widths[c] = max(col_widths[c],len(col))
    template = '{{:{:d}s}} = ({{:{:d}s}} +/- {{:{:d}s}}) {{:{:d}s}} {{}}'.format(*[c+1 for c in col_widths])
    header = template.format('Qualifier','Mean','Std','50%','Unit')
    print("SUMMARY TABLE FROM MARGINALISED TRACES")
    print('='*len(header))
    print(header)
    print('='*len(header))
    for line in table:
        print(template.format(*line))
    print('='*len(header))
    
    if context=='lmfit':
        print("\nSUMMARY TABLE FROM FITTER")
        print('='*len(header))
        print(header)
        print('='*len(header))
        for line in table_from_fitter:
            print(template.format(*line))
        print('='*len(header))
    
    #-- now for the correlations
    for i,(ipar,iind,ipath) in enumerate(zip(had,indices,paths)):
        for j,(jpar,jind,jpath) in enumerate(zip(had,indices,paths)):
            if j<=i: continue
            plt.figure()
            plt.xlabel("/".join(ipath)+' [{}]'.format(units[i]))
            plt.ylabel("/".join(jpath)+' [{}]'.format(units[j]))
            if correlations=='hexbin':
                plt.hexbin(fitparams['feedback']['traces'][iind],
                           fitparams['feedback']['traces'][jind],bins='log',gridsize=50)
                cbar = plt.colorbar()
                cbar.set_label("Number of occurrences")
            elif correlations=='walk':
                points = np.array([fitparams['feedback']['traces'][iind],
                           fitparams['feedback']['traces'][jind]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                N = float(len(points))
                lc = LineCollection(segments, cmap=plt.get_cmap('Greys'),
                    norm=plt.Normalize(-0.1*N, N))
                lc.set_array(np.arange(len(points)))
                lc.set_linewidth(2)
                plt.gca().add_collection(lc)
                plt.grid()
                plt.xlim(points[:,0,0].min(),points[:,0,0].max())
                plt.ylim(points[:,0,1].min(),points[:,0,1].max())
                cbar = plt.colorbar(lc)
                cbar.set_label("Iteration number")
                
    print("MCMC path length = {}".format(len(fitparams['feedback']['traces'][iind])))
    return None


def accept_fit(system,fitparams):
    """
    Accept the fit.
    """
    feedback = fitparams['feedback']
    fitmethod = fitparams.context.split(':')[1]
    #-- which parameters were fitted?
    fitted_param_labels = [par.get_unique_label() for par in feedback['parameters']]
    #-- walk through all the parameterSets available:
    #walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
    for parset in system.walk():
        
        # If the parameterSet is not enabled, skip it
        if not parset.get_enabled():
            continue
        
        #-- fore ach parameterSet, walk to all the parameters
        for qual in parset:
            #-- extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                this_param = parset.get_parameter(qual)
                #-- ask a unique ID and update the value of the parameter
                myid = this_param.get_unique_label()
                index = fitted_param_labels.index(myid)
                #-- [iwalker,trace,iparam]
                if fitmethod in ['pymc','emcee']:
                    best_index = np.argmax(np.array(feedback['lnprobability']))
                    this_param.set_value(feedback['traces'][index][best_index])
                    try:
                        this_param.set_posterior(feedback['traces'][index],update=False)
                    except:
                        this_param.set_posterior(distribution='sample',sample=feedback['traces'][index],discrete=False)
                    logger.info("Set {} = {} (and complete posterior) from MCMC trace".format(qual,this_param.as_string()))
                elif fitmethod in ['lmfit']:
                    this_param.set_value(feedback['values'][index])
                    this_param.set_posterior(distribution='normal',
                                             mu=feedback['values'][index],
                                             sigma=feedback['sigmas'][index])
                    logger.info("Set {} = {} from {} fit".format(qual,this_param.as_string(),fitparams['method']))
                else:
                    logger.info("Did not recognise fitting method {}".format(fitmethod))



def longitudinal_field(times,Bl=None,rotperiod=1.,
                  Bpolar=100.,ld_coeff=0.3,beta=90.,incl=90.,phi0=0.,
                  fix=('rotperiod','ld_coeff')):
    """
    Trial function to derive parameters from Stokes V profiels.
    
    See e.g. [Preston1967]_ or [Donati2001]_.
    """
    
    def residual(pars,times,Bl=None):
        rotperiod = pars['rotperiod'].value
        Bp = pars['Bpolar'].value
        u = pars['ld_coeff'].value
        beta = pars['beta'].value/180.*np.pi
        incl = pars['incl'].value/180.*np.pi
        phi0 = pars['phi0'].value
        phases = (times%rotperiod)/rotperiod
    
        model = Bp*(15.+u)/(20.*(3.-u)) * (np.cos(beta)*np.cos(incl) + np.sin(beta)*np.sin(incl)*np.cos(2*np.pi*(phases-phi0)))
        
        if Bl is None:
            return model
        else:
            return (Bl-model)
    
    params = lmfit.Parameters()
    params.add('rotperiod',value=rotperiod,vary=('rotperiod' not in fix))
    params.add('Bpolar',value=Bpolar,vary=('Bpolar' not in fix))
    params.add('ld_coeff',value=ld_coeff,vary=('ld_coeff' not in fix),min=0,max=1)
    params.add('beta',value=beta,vary=('beta' not in fix),min=-180,max=180)
    params.add('incl',value=incl,vary=('incl' not in fix),min=-180,max=180)
    params.add('phi0',value=phi0,vary=('phi0' not in fix),min=0,max=1)
    
    if Bl is not None:
        out = lmfit.minimize(residual,params,args=(times,Bl))
        lmfit.report_errors(params)
        for par in params:
            params[par].min = None
            params[par].max = None
        out = lmfit.minimize(residual,params,args=(times,Bl))
    
        lmfit.report_errors(params)
    
    final_model = residual(params,times)
    
    if Bl is None:
        return final_model
    else:
        return params, final_model
        
        
        
class UserCosts(object):
    """
    [FUTURE]
    This class allows the user an interface for creating their own penalties to the cost-function used in fitting.
    
    In order to include your own functions, you must subclass this class, write function(s) using the correct format,
    and send your class initialized with the function names you'd like the cost function to call.
    
    NOTE: this currently only works for fitting:emcee and will be IGNORED for all other fitting types
    
    Example
    import numpy as np
    from phoebe.backend.fitting import UserCost:
    class AdditionalCosts(UserCost):
        def my_cost(self, system, log_f, chi2, n_data):
            if np.random.random() > 0.5:
                log_f = -np.inf
                chi2.append(np.inf)
                n_data += 1
            return log_f, chi2, n_data
            
    additional_funcs = AdditionalCosts(['my_cost'])
    
    """
    def __init__(self, funcnames=[]):
        """
        [FUTURE]
        set the functions that need to be called in the cost function
        """
        if isinstance(funcnames, str):
            funcnames = [funcnames]
        self.funcs = funcnames
        
    def run(self, system, log_f, chi2, n_data):
        """
        [FUTURE]
        run each of the desired functions.  Each overwrites the previous values for
        log_f, chi2, n_data and will return the resulting values after looping over
        all functions.
        
        This function shouldn't be called by the user, but will be called by 
        BodyBag.get_logp
        """
        for func in self.funcs:
            log_f, chi2, n_data = getattr(self, func)(system, log_f, chi2, n_data)
            
        return log_f, chi2, n_data
        
    def example_cost_function(self, system, log_f, chi2, n_data):
        """
        [FUTURE]
        This is an example of a cost-function which could be created by the user.
        Notice that the input arguments must be self, system, log_f, chi2, n_data.
        
        In this case those values are penalized randomly, but in either case the final
        values for log_f, chi2, n_data are returned by the function.
        """
        
        if np.random.random() > 0.5:
            log_f = -np.inf
            chi2.append(np.inf)
            n_data += 1
        
        return log_f, chi2, n_data
        

#if __name__=="__main__":
    #import sys
    ## retrieve info and pickles
    #fitter,system_file,params_file,mpi_file,fitparams_file = sys.argv[1:]
    
    #if fitter=='run_emcee':
        #pool = MPIPool()
    
        #if not pool.is_master():
            #pool.wait()
            #sys.exit(0)
    
        #system = universe.load(system_file)
        #if os.path.isfile(params_file):
            #params = parameters.load(params_file)
        #else:
            #params = None
        #if os.path.isfile(mpi_file):
            #mpi = parameters.load(mpi_file)
        #else:
            #mpi = None
        #if os.path.isfile(fitparams_file):
            #fitparams = parameters.load(fitparams_file)
        #else:
            #fitparams = None
        
        #fitparams = run_emcee(system,params,mpi,fitparams,pool=pool)
        
        #pool.close()
    
        #fitparams.save('testiewestie')
