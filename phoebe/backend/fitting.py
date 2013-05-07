"""
Fitting routines, minimization, optimization.

Section 1. Summary
==================

The top level function you should use is L{run}.

**Main functions**

.. autosummary::

   run
   accept_fit
   summarize_fit
   
**Individual fitting routines**

.. autosummary::

   run_pymc
   run_emcee
   run_lmfit
   run_grid
   run_minuit
   
**Wilson Devinney**

.. autosummary::
   
   run_wd_emcee
   
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
import time
import logging
import functools
import itertools
import copy
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from phoebe.utils import utils
from phoebe.backend import universe
from phoebe.backend import decorators
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



def run(system,params=None,fitparams=None,mpi=None,accept=False):
    """
    Run a fitting routine.
    
    The type of code to use is defined by the context of C{fitparams}:
    
        1. :ref:`fitting:pymc <parlabel-phoebe-fitting:pymc>`: Metropolis-Hastings MCMC via pymc
        2. :ref:`fitting:emcee <parlabel-phoebe-fitting:emcee>`: Affine Invariant MCMC via emcee
        3. :ref:`fitting:lmfit <parlabel-phoebe-fitting:lmfit>`: nonlinear optimizers via lmfit
        4. :ref:`fitting:minuit <parlabel-phoebe-fitting:minuit>`: nonlinear optimizers via MINUIT
        5. :ref:`fitting:grid <parlabel-phoebe-fitting:grid>`: grid computations
    
    The parameters determining how the system is computed are defined by the
    ParameterSet :ref:`params <parlabel-phoebe-compute>`.
    
    The computations can be threaded if an :ref:`mpi <parlabel-phoebe-mpi>`
    ParameterSet is supplied.
    
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
    
    mylogger = logging.getLogger("")
    for handler in mylogger.handlers:
        if not hasattr(handler,'baseFilename'):
            if mylogger.level<logging._levelNames['WARNING']:
                handler.setLevel('WARNING')
    utils.add_filehandler(mylogger,flevel='INFO',filename='fitting_{}.log'.format("_".join(time.asctime().split())))
    
    #-- figure out the solver, and run it
    if params is None:
        params = parameters.ParameterSet(context='compute')
        logger.info("No compute parameters given: adding default set")
    #-- default fitter is nelder
    if fitparams is None:
        fitparams = parameters.ParameterSet(frame='phoebe',context='fitting:lmfit')
    if fitparams.context=='fitting:pymc':
        solver = run_pymc
    elif fitparams.context=='fitting:emcee':
        #-- we'll set a default fitparams here because of the MPIRUN
        if fitparams is None:
            fitparams = parameters.ParameterSet(frame='phoebe',context='fitting:emcee',walkers=20,iters=10)
        solver = run_emcee
    elif fitparams.context=='fitting:lmfit':
        solver = run_lmfit
    elif fitparams.context=='fitting:minuit':
        solver = run_minuit
    elif fitparams.context=='fitting:grid':
        solver = run_grid
    else:
        raise NotImplementedError("Fitting context {} is not understood".format(fitparams.context))
    fitparams = solver(system,params=params,mpi=mpi,fitparams=fitparams)
    
    #-- reset the logger
    mylogger.handlers = mylogger.handlers[:-1]
    for handler in mylogger.handlers:
        if not hasattr(handler,'baseFilename'):
            if mylogger.level<logging._levelNames['WARNING']:
                handler.setLevel('INFO')
    logger.info("Reset logger")
    
    #-- accept the best fitting model and compute the system
    if accept:
        accept_fit(system,fitparams)
        system.reset()
        system.clear_synthetic()
        try:
            system.compute(params=params,mpi=mpi)
        except Exception,msg:
            print system.params.values()[0]
            logger.info("Could not accept for some reason (original message: {})".format(msg))
    
    return fitparams


                
            





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
                if myid in ids: continue
                #-- else, add the name to the list of pnames. Ask for the
                #   prior of this object
                name = '{}_{}'.format(qual,myid)
                pars[name] = parameter.get_prior(name=name,fitter='pymc')
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
        system.compute(params=params,mpi=mpi)
        mu,sigma,model = system.get_model()
        return model
   
    mu,sigma,model = system.get_model()
    #mu,sigma,model = system.get_data()
    
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

#@decorators.mpirun_emcee
def run_emcee(system,params=None,mpi=None,fitparams=None,pool=None):
    """
    Perform MCMC sampling of the parameter space of a system using emcee.
    
    Be sure to set the parameters to fit to be adjustable in the system.
    Be sure to set the priors of the parameters to fit in the system.
    
    **Tips**
    
    * Have a look at the C{acortime}, it is good practice to let the sampler
      run at least 10 times the C{acortime}. If ``acortime = np.nan``, you
      probably need to take more iterations!
    * Use as many walkers as possible (hundreds for such a handful of
      parameters)
    * Beware of a burn in period. The most conservative you can get is making
      a histogram of only the final states of all walkers.
    * To get the original chains per walker back, and make a histogram of the
      final states, do
    
    >>> chains = fitparams['feedback']['traces'][0].reshape((walkers,-1))
    
    Then you should be able to recover a sampling of the prior via
    
    >>> freqs,bins = np.hist(chains[0])
    
    And (hopefully) of the posterior via
    
    >>> freqs,bins = np.hist(chains[-1])
    
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
    if fitparams is None:
        fitparams = parameters.ParameterSet(frame='phoebe',context='fitting:emcee',walkers=20,iters=10)
    nwalkers = fitparams['walkers']
    # We need unique names for the parameters that need to be fitted, we need
    # initial values and identifiers to distinguish parameters with the same
    # name (we'll also use the identifier in the parameter name to make sure
    # they are unique). While we are iterating over all the parameterSets,
    # we'll also have a look at what context they are in. From this, we decide
    # which fitting algorithm to use.
    ids = []
    pars = []
    names = []
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
                if myid in ids: continue
                #-- and add the id
                ids.append(myid)
                pars.append(parameter.get_value_from_prior(size=nwalkers))
                names.append(qual)
    pars = np.array(pars).T
    #-- now, if the number of walkers is smaller then twice the number of
    #   parameters, adjust that number to the required minimum and raise a
    #   warning
    if (2*pars.shape[1])>nwalkers:
        logger.warning("Number of walkers ({}) cannot be smaller than 2 x npars: set to {}".format(nwalkers,2*pars.shape[1]))
        nwalkers = 2*pars.shape[1]
        fitparams['walkers'] = nwalkers
    #-- derive which algorithm to use for fitting. If all the contexts are the
    #   same, it's easy. Otherwise, it's ambiguous and we raise a ValueError
    algorithm = set(frames)
    if len(algorithm)>1:
        raise ValueError("Ambiguous set of parameters (different frames, found): {}".format(algorithm))
    else:
        algorithm = list(algorithm)[0]
        logger.info('Choosing back-end {}'.format(algorithm))
    
    def lnprob(pars,ids,system):
        #-- evaluate the system, get the results and return a probability
        had = []
        any_outside_limits = False
        #-- walk through all the parameterSets available:
        walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
        
        try:
            for parset in walk:
                #-- for each parameterSet, walk to all the parameters
                for qual in parset:
                    #-- extract those which need to be fitted
                    if parset.get_adjust(qual) and parset.has_prior(qual):
                        #-- ask a unique ID and update the value of the parameter
                        this_param = parset.get_parameter(qual)
                        myid = this_param.get_unique_label()
                        if myid in had: continue
                        index = ids.index(myid)
                        parset[qual] = pars[index]
                        had.append(myid)
                        #-- if this parameter is outside the limits, we know
                        #   the model is crap and forget about it immediately
                        if not this_param.is_inside_limits():
                            any_outside_limits = True
                            raise StopIteration
        #-- if any of the parameters is outside the bounds, we don't really
        #   compute the model
        except StopIteration:
            logger.warning("At least one of the parameters was outside bounds")
            return -np.inf
        
        system.reset()
        system.clear_synthetic()
        system.compute(params=params,mpi=mpi)
        logp,chi2,N = system.get_logp()
        #mu,sigma,model = system.get_model()
        return logp
    
    #-- if we need to do incremental stuff, we'll need to open a chain file
    #   if we don't have feedback already
    start_iter = 0
    if fitparams['incremental'] and not fitparams['feedback']:
        chain_file = 'emcee_chain.{}'.format(fitparams['label'])
        #-- if the file already exists, choose the starting position to be
        #   the last position from the file
        if os.path.isfile(chain_file):
            f = open(chain_file,'r')
            last_lines = [line.strip().split() for line in f.readlines()[-nwalkers:]]
            start_iter = int(np.array(last_lines[-1])[0])+1
            pars = np.array(last_lines,float)[:,2:]         
            f.close()
            logger.info("Starting from previous EMCEE chain from file")
        else:
            logger.info("Starting new EMCEE chain")
        f = open(chain_file,'a')
    #-- if we do have feedback already we can simply load the final state
    #   from the feedback
    elif fitparams['incremental']:
        pars = np.array(fitparams['feedback']['traces'])
        pars = pars.reshape(pars.shape[0],-1,nwalkers)
        pars = pars[:,-1,:].T
        start_iter = len(fitparams['feedback']['traces'][0])/nwalkers
        logger.info("Starting from previous EMCEE chain from feedback")
    #-- run the sampler
    sampler = emcee.EnsembleSampler(nwalkers,pars.shape[1],lnprob,
                                    args=[ids,system],
                                    threads=fitparams['threads'],pool=pool)
    num_iterations = fitparams['iters']-start_iter
    if num_iterations<=0:
        logger.info('EMCEE contains {} iterations already (iter={})'.format(start_iter,fitparams['iters']))
        num_iterations=0
    logger.warning("EMCEE: varying parameters {}".format(', '.join(names)))
    for i,result in enumerate(sampler.sample(pars,iterations=num_iterations,storechain=True)):
        niter = i+start_iter
        logger.info("EMCEE: Iteration {} of {} ({:.3f}% complete)".format(niter,fitparams['iters'],float(niter)/fitparams['iters']*100.))
        #-- if we want to keep track of incremental stuff, write each iteratiion
        #   to a file
        if fitparams['incremental'] and not fitparams['feedback']:
            position = result[0]
            for k in range(position.shape[0]):
                values = " ".join(["{:.16e}".format(l) for l in position[k]])
                f.write("{0:9d} {1:9d} {2:s}\n".format(niter,k,values))
            f.flush()
    if fitparams['incremental'] and not fitparams['feedback']:
        f.close()
    
    #-- acceptance fraction should be between 0.2 and 0.5 (rule of thumb)
    logger.info("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    
    #-- store the info in a feedback dictionary
    try:
        acortime = sampler.acor
    #RuntimeError: The autocorrelation time is too long relative to the variance in dimension 
    except RuntimeError:
        acortime = np.nan
        logger.warning('Probably not enough iterations for emcee')
    feedback = dict(parset=fitparams,parameters=[],traces=[],priors=[],accfrac=np.mean(sampler.acceptance_fraction),
                    lnprobability=sampler.flatlnprobability,acortime=acortime)
    
    #-- add the posteriors to the parameters
    had = []
    #-- walk through all the parameterSets available:
    walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
    for parset in walk:
        #-- fore ach parameterSet, walk to all the parameters
        for qual in parset:
            #-- extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                #-- ask a unique ID and update the value of the parameter
                myid = parset.get_parameter(qual).get_unique_label()
                if myid in had: continue
                had.append(myid)
                index = ids.index(myid)
                #-- [iwalker,trace,iparam]
                trace = sampler.flatchain[:,index]
                this_param = parset.get_parameter(qual)
                #this_param.set_posterior(trace.ravel())
                feedback['parameters'].append(this_param)
                feedback['traces'].append(trace)
                feedback['priors'].append(this_param.get_prior(fitter=None))
    if fitparams['feedback'] and fitparams['incremental']:
        feedback['traces'] = np.hstack([fitparams['feedback']['traces'],feedback['traces']])
        feedback['lnprobability'] = np.hstack([fitparams['feedback']['lnprobability'],feedback['lnprobability']])
    fitparams['feedback'] = feedback
    return fitparams



#}

#{ Nonlinear optimizers

def run_lmfit(system,params=None,mpi=None,fitparams=None):
    """
    Iterate an lmfit to start from different representations.
    """
    if fitparams['iters']>0:
        set_values_from_prior = True
    else:
        set_values_from_prior = False
    iters = max(fitparams['iters'],1)
    best_chi2 = np.inf
    best_fitparams = None
    for iteration in range(iters):
        logger.warning("Iteration {}/{}".format(iteration+1,iters))
        if set_values_from_prior:
            system.set_values_from_priors()
        fitparams = _run_lmfit(system,params=params,mpi=mpi,fitparams=fitparams)
        if fitparams['feedback']['redchi']<best_chi2:
            best_fitparams = fitparams
            logger.warning("Found better fit = {} < {}".format(fitparams['feedback']['redchi'],best_chi2))
            best_chi2 = fitparams['feedback']['redchi']
            
    return best_fitparams


def _run_lmfit(system,params=None,mpi=None,fitparams=None):
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
    if fitparams is None:
        fitparams = parameters.ParameterSet(frame='phoebe',context='fitting:lmfit')
    # We need unique names for the parameters that need to be fitted, we need
    # initial values and identifiers to distinguish parameters with the same
    # name (we'll also use the identifier in the parameter name to make sure
    # they are unique). While we are iterating over all the parameterSets,
    # we'll also have a look at what context they are in. From this, we decide
    # which fitting algorithm to use.
    ids = []
    pars = lmfit.Parameters()
    ppars = [] # Phoebe parameters
    #-- walk through all the parameterSets available. This needs to be via
    #   this utility function because we need to iteratively walk down through
    #   all BodyBags too.
    #walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
    frames = []
    for parset in system.walk():#walk:
        frames.append(parset.frame)
        #-- for each parameterSet, walk through all the parameters
        for qual in parset:
            #-- extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                #-- ask a unique ID and check if this parameter has already
                #   been treated. If so, continue to the next one.
                parameter = parset.get_parameter(qual)
                myid = parameter.get_unique_label()
                if myid in ids: continue
                #-- else, add the name to the list of pnames. Ask for the
                #   prior of this object
                name = '{}_{}'.format(qual,myid).replace('-','_')
                prior = parameter.get_prior()
                if fitparams['bounded']:
                    minimum,maximum = prior.get_limits()
                else:
                    minimum,maximum = None,None
                pars.add(name,value=parameter.get_value(),min=minimum,max=maximum,vary=True)
                #-- and add the id
                ids.append(myid)
                ppars.append(parameter)
    #-- derive which algorithm to use for fitting. If all the contexts are the
    #   same, it's easy. Otherwise, it's ambiguous and we raise a ValueError
    #algorithm = set(frames)
    #if len(algorithm)>1:
        #raise ValueError("Ambiguous set of parameters (different frames, found): {}".format(algorithm))
    #else:
        #algorithm = list(algorithm)[0]
        #logger.info('Choosing back-end {}'.format(algorithm))
    
    traces = []
    redchis = []
    Nmodel = dict()
    
    def model_eval(pars,system):
        #-- evaluate the system, get the results and return a probability
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
                    parset[qual] = pars['{}_{}'.format(qual,myid)].value
                    had.append(myid)
        
        system.reset()
        system.clear_synthetic()
        system.compute(params=params,mpi=mpi)
        mu,sigma,model = system.get_model()
        retvalue = (model-mu)/sigma
        
        #-- short log message:
        names = [par for par in pars]
        vals = [pars[par].value for par in pars]
        logger.warning("Current values: {} (chi2={:.6g})".format(", ".join(['{}={}'.format(name,val) for name,val in zip(names,vals)]),(retvalue**2).mean()))
        #-- keep track of trace
        traces.append(vals)
        redchis.append(np.array(retvalue**2).sum()/(len(model)-len(pars)))
        Nmodel['Nd'] = len(model)
        Nmodel['Np'] = len(pars)
        return retvalue
    
    #-- do the fit and report on the errors to the screen
    #   we suffer quite drastically from numerical effects, i.e. that for
    #   example the total integrated flux changes relatively drastically for
    #   very small inclination changes. This messes up the leastsq algorithm,
    #   so we need to tell it that it has to take wide steps for the computation
    #   of its derivatives.
    extra_kwargs = {}
    if fitparams['method']=='leastsq':
        extra_kwargs['epsfcn'] = 1e-3
    result = lmfit.minimize(model_eval,pars,args=(system,),method=fitparams['method'],**extra_kwargs)
    lmfit.report_errors(pars)
    #-- extract the values to put them in the feedback
    if not result.success:
        logger.error("nonlinear fit with method {} failed".format(fitparams['method']))
        values = [np.nan for ipar in pars]
    else:
        values = [pars['{}_{}'.format(ipar.get_qualifier(),ipar.get_unique_label().replace('-','_'))].value for ipar in ppars]
    #-- the same with the errors and correlation coefficients, if there are any
    if result.errorbars:
        sigmas = [pars['{}_{}'.format(ipar.get_qualifier(),ipar.get_unique_label().replace('-','_'))].stderr for ipar in ppars]
        correl = [pars['{}_{}'.format(ipar.get_qualifier(),ipar.get_unique_label().replace('-','_'))].correl for ipar in ppars]
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
            ci = lmfit.conf_interval(result,sigmas=(0.674,0.997),verbose=True,maxiter=10)
            lmfit.printfuncs.report_ci(ci)
        except Exception,msg:
            logger.error("Could not estimate CI (original error: {}".format(str(msg)))
    feedback = dict(parameters=ppars,values=values,sigmas=sigmas,correls=correl,\
                    redchi=result.redchi,success=result.success,
                    traces=np.array(traces).T,redchis=redchis,
                    Ndata=Nmodel['Nd'],Npars=Nmodel['Np'])
    fitparams['feedback'] = feedback
    return fitparams


class MinuitMinimizer(object):
    def __init__(self,system,params=None,mpi=None):
        self.system = system
        self.params = params # computational parameters
        self.mpi = mpi
        
        #-- evaluate the system, get the results and return a probability
        had = []
        #-- walk through all the parameterSets available:
        walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
        self.param_names = []
        for parset in walk:
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
        walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
        for parset in walk:
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
    walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
    for parset in walk:
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
                    lower,upper = this_param.get_prior().get_limits()
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
    print dir(m)
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
    # We need unique names for the parameters that need to be fitted, we need
    # initial values and identifiers to distinguish parameters with the same
    # name (we'll also use the identifier in the parameter name to make sure
    # they are unique). While we are iterating over all the parameterSets,
    # we'll also have a look at what context they are in. From this, we decide
    # which fitting algorithm to use.
    ids = []
    names = []
    ranges = []
    #-- walk through all the parameterSets available. This needs to be via
    #   this utility function because we need to iteratively walk down through
    #   all BodyBags too.
    #walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
    for parset in system.walk():
        #-- for each parameterSet, walk through all the parameters
        for qual in parset:
            #-- extract those which need to be fitted
            if parset.get_adjust(qual) and parset.has_prior(qual):
                #-- ask a unique ID and check if this parameter has already
                #   been treated. If so, continue to the next one.
                parameter = parset.get_parameter(qual)
                myid = parameter.get_unique_label()
                if myid in ids: continue
                #-- construct the range:
                myrange = parameter.get_prior().distr_pars['bins']
                #-- and add the id
                ids.append(myid)
                names.append(qual)
                ranges.append(myrange) 
    
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
        logger.warning('GRID: step {} - parameters: {}'.format(i,msg))
        mylogp = lnprob(pars,ids,system)
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

#{ Verifiers

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
    walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
    for parset in walk:
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
                    logger.info("Set {} = {} from {} fit".format(qual,this_param.as_string(),fitparams['method']))
                else:
                    logger.info("Did not recognise fitting method {}".format(fitmethod))



def longitudinal_field(times,Bl,rotperiod=1.,
                  Bpolar=100.,ld_coeff=0.3,beta=90.,incl=90.,phi0=0.,
                  fix=('rotperiod','ld_coeff')):
    """
    Trial function to derive parameters from Stokes V profiels.
    
    See e.g. [Preston1967]_ or [Donati2001]_.
    """
    
    def residual(pars,times,Bl):
        rotperiod = pars['rotperiod'].value
        Bp = pars['Bpolar'].value
        u = pars['ld_coeff'].value
        beta = pars['beta'].value/180.*np.pi
        incl = pars['incl'].value/180.*np.pi
        phi0 = pars['phi0'].value
        phases = (times%rotperiod)/rotperiod
    
        model = Bp*(15.+u)/(20.*(3.-u)) * (np.cos(beta)*np.cos(incl) + np.sin(beta)*np.sin(incl)*np.cos(2*np.pi*(phases-phi0)))
        
        return (Bl-model)
    
    params = lmfit.Parameters()
    params.add('rotperiod',value=rotperiod,vary=('rotperiod' not in fix))
    params.add('Bpolar',value=Bpolar,vary=('Bpolar' not in fix))
    params.add('ld_coeff',value=ld_coeff,vary=('ld_coeff' not in fix),min=0,max=1)
    params.add('beta',value=beta,vary=('beta' not in fix),min=-180,max=180)
    params.add('incl',value=incl,vary=('incl' not in fix),min=-180,max=180)
    params.add('phi0',value=phi0,vary=('phi0' not in fix),min=0,max=1)
    
    
    out = lmfit.minimize(residual,params,args=(times,Bl))
    lmfit.report_errors(params)
    for par in params:
        params[par].min = None
        params[par].max = None
    out = lmfit.minimize(residual,params,args=(times,Bl))
    
    lmfit.report_errors(params)
    
    return params
    
    
    



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