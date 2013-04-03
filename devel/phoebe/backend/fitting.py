"""
Fitting routines, minimization, optimization.

The top level function you should use is L{run}.
"""
import os
import time
import logging
import functools
import itertools
import copy
import numpy as np
from matplotlib import pyplot as plt
from phoebe.utils import utils
from phoebe.backend import universe
from phoebe.backend import observatory
from phoebe.wd import wd
from phoebe.parameters import parameters
try:
    import pymc
except ImportError:
    print("Unable to load mcmc fitting routines from pymc: restricted fitting facilities")
try:
    import emcee
except ImportError:
    print("Unable to load mcmc fitting routines from emcee: restricted fitting facilities")
try:
    import lmfit
except ImportError:
    print("Unable to load nonlinear fitting routines from lmfit: restricted fitting facilities")
    
logger = logging.getLogger("FITTING")



def run(system,params=None,fitparams=None,mpi=None,accept=False):
    """
    Run a fitting routine.
    
    The type of code to use is defined by the context of C{fitparams}:
    
        1. :ref:`fitting:pymc <parlabel-phoebe-fitting:pymc>`: Metropolis-Hastings MCMC via pymc
        2. :ref:`fitting:emcee <parlabel-phoebe-fitting:emcee>`: Affine Invariant MCMC via emcee
        3. :ref:`fitting:lmfit <parlabel-phoebe-fitting:lmfit>`: nonlinear optimizers via lmfit
    
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
        solver = run_emcee
    elif fitparams.context=='fitting:lmfit':
        solver = run_lmfit
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
        observatory.compute(system,params=params,mpi=mpi)
    
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
        observatory.compute(system,params=params,mpi=mpi)
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


def run_emcee(system,params=None,mpi=None,fitparams=None):
    """
    Perform MCMC sampling of the parameter space of a system using emcee.
    
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
    @return: the MCMC sampling history (ParameterSet of context 'fitting:emcee'
    @rtype: ParameterSet
    """
    if fitparams is None:
        fitparams = parameters.ParameterSet(frame='phoebe',context='fitting:emcee')
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
                    index = ids.index(myid)
                    parset[qual] = pars[index]
                    had.append(myid)
        system.reset()
        system.clear_synthetic()
        observatory.compute(system,params=params,mpi=mpi)
        logp = system.get_logp()
        mu,sigma,model = system.get_model()
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
        start_iter = len(fitparams['feedback']['traces'][0])
        logger.info("Starting from previous EMCEE chain from feedback")
    #-- run the sampler
    sampler = emcee.EnsembleSampler(nwalkers,pars.shape[1],lnprob,args=[ids,system],threads=fitparams['threads'])
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
    feedback = dict(parset=fitparams,parameters=[],traces=[],priors=[],accfrac=np.mean(sampler.acceptance_fraction))
    
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
    fitparams['feedback'] = feedback
    return fitparams



def run_wd_emcee(system,data,fitparams=None):
    """
    Perform MCMC sampling of the WD parameter space of a system using emcee.
    
    Be sure to set the parameters to fit to be adjustable in the system.
    Be sure to set the priors of the parameters to fit in the system.
    
    @param system: the system to fit
    @type system: 3xtuple (ps,lc,rv)
    @param data: the data to fit
    @type data: 2xtuple (lcobs,rvobs)
    @param fitparams: fit algorithm parameters
    @type fitparams: ParameterSet
    @return: the MCMC sampling history
    """
    from phoebe.wd import wd
    if fitparams is None:
        fitparams = parameters.ParameterSet(frame='phoebe',context='fitting:mcmc')
    nwalkers = fitparams['walkers']
    # We need unique names for the parameters that need to be fitted, we need
    # initial values and identifiers to distinguish parameters with the same
    # name (we'll also use the identifier in the parameter name to make sure
    # they are unique). While we are iterating over all the parameterSets,
    # we'll also have a look at what context they are in. From this, we decide
    # which fitting algorithm to use.
    ids = []
    pars = []
    #-- walk through all the parameterSets available. This needs to be via
    #   this utility function because we need to iteratively walk down through
    #   all BodyBags too.
    frames = []
    for parset in system:
        #-- for each parameterSet, walk through all the parameters
        for qual in parset:
            #-- extract those which need to be fitted
            if parset.get_adjust(qual):
                #-- ask a unique ID and check if this parameter has already
                #   been treated. If so, continue to the next one.
                parameter = parset.get_parameter(qual)
                myid = parameter.get_unique_label()
                if myid in ids: continue
                #-- and add the id
                ids.append(myid)
                pars.append(parameter.get_value_from_prior(size=nwalkers))
    pars = np.array(pars).T
    #-- now, if the number of walkers is smaller then twice the number of
    #   parameters, adjust that number to the required minimum and raise a
    #   warning
    if (2*pars.shape[1])<nwalkers:
        logger.warning("Number of walkers ({}) cannot be smaller than 2 x npars: set to {}".format(nwalkers,2*pars.shape[1]))
        nwalkers = 2*pars.shape[1]
        fitparams['walkers'] = nwalkers
    def lnprob(pars,ids,system):
        #-- evaluate the system, get the results and return a probability
        had = []
        #-- walk through all the parameterSets available:
        for parset in system:
            #-- for each parameterSet, walk to all the parameters
            for qual in parset:
                #-- extract those which need to be fitted
                if parset.get_adjust(qual):
                    #-- ask a unique ID and update the value of the parameter
                    myid = parset.get_parameter(qual).get_unique_label()
                    if myid in had: continue
                    index = ids.index(myid)
                    parset[qual] = pars[index]
                    had.append(myid)
        output,params = wd.lc(system[0],light_curve=system[1],rv_curve=system[2])
        logp = 0
        for dtype in ['lc','rv']:
            term1 = - 0.5*np.log(2*np.pi*sigma**2)
            term2 = - (data['lc']-output['lc'])**2/(2.*sigma**2)
            logp = logp + (term1 + term2).mean()
            logp = logp + (term1 + term2).sum()
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
        start_iter = len(fitparams['feedback']['traces'][0])
        logger.info("Starting from previous EMCEE chain from feedback")
    #-- run the sampler
    sampler = emcee.EnsembleSampler(nwalkers,pars.shape[1],lnprob,args=[ids,system])
    num_iterations = fitparams['iters']-start_iter
    if num_iterations<=0:
        logger.info('EMCEE contains {} iterations already (iter={})'.format(start_iter,fitparams['iter']))
        num_iterations=0
    for i,result in enumerate(sampler.sample(pars,iterations=num_iterations,storechain=True)):
        niter = i+start_iter
        logger.info("EMCEE: Iteration {} of {} ({:.3f}% complete)".format(niter,fitparams['iter'],float(niter)/fitparams['iter']*100.))
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
    feedback = dict(parset=fitparams,parameters=[],traces=[],priors=[],accfrac=np.mean(sampler.acceptance_fraction))
    
    #-- add the posteriors to the parameters
    had = []
    #-- walk through all the parameterSets available:
    walk = utils.traverse(system,list_types=(universe.BodyBag,universe.Body,list,tuple),dict_types=(dict,))
    for parset in walk:
        #-- fore ach parameterSet, walk to all the parameters
        for qual in parset:
            #-- extract those which need to be fitted
            if parset.get_adjust(qual):
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
                name = '{}_{}'.format(qual,myid).replace('-','_')
                prior = parameter.get_prior()
                if fitparams['bounded']:
                    minimum,maximum = prior['lower'],prior['upper']
                else:
                    minimum,maximum = None,None
                pars.add(name,value=parameter.get_value(),min=minimum,max=maximum,vary=True)
                #-- and add the id
                ids.append(myid)
                ppars.append(parameter)
    #-- derive which algorithm to use for fitting. If all the contexts are the
    #   same, it's easy. Otherwise, it's ambiguous and we raise a ValueError
    algorithm = set(frames)
    if len(algorithm)>1:
        raise ValueError("Ambiguous set of parameters (different frames, found): {}".format(algorithm))
    else:
        algorithm = list(algorithm)[0]
        logger.info('Choosing back-end {}'.format(algorithm))
    
    
    def model_eval(pars,system):
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
                    myid = parset.get_parameter(qual).get_unique_label().replace('-','_')
                    if myid in had: continue
                    parset[qual] = pars['{}_{}'.format(qual,myid)].value
                    had.append(myid)
        system.reset()
        system.clear_synthetic()
        observatory.compute(system,params=params,mpi=mpi)
        mu,sigma,model = system.get_model()
        retvalue = (model-mu)/sigma
        
        #-- short log message:
        names = [par for par in pars]
        vals = [pars[par].value for par in pars]
        #logger.warning("Current values: {} (chi2={:.6g})".format(", ".join(['{}={}'.format(name,val) for name,val in zip(names,vals)]),(retvalue**2).mean()))
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
                    redchi=result.redchi,success=result.success)
    fitparams['feedback'] = feedback
    return fitparams


#}

#{ Gridding


def run_grid(system,params=None,mpi=None,fitparams=None):
    """
    Compute a grid in the adjustable parameter space.
    
    Be sure to set the parameters to grid to be adjustable in the system.
    Be sure to set the limits and step of the parameters to grid in the system.
    
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
                #-- construct the range:
                myrange = parameter.get_prior(fitter=None)['range']
                #-- and add the id
                ids.append(myid)
                names.append(qual)
                ranges.append(myrange)
                
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
                    index = ids.index(myid)
                    parset[qual] = pars[index]
                    had.append(myid)
        system.reset()
        system.clear_synthetic()
        observatory.compute(system,params=params,mpi=mpi)
        logp = system.get_logp()
        return logp
    
    #-- now run over the whole grid
    grid_pars = []
    grid_logp = []
    for i,pars in enumerate(itertools.product(*ranges)):
        mylogp = lnprob(pars,ids,system)
        this_system = copy.deepcopy(system)
        this_system.remove_mesh()
        this_system.save('gridding_{:05d}.phoebe'.format(i))
        del this_system
        grid_pars.append(pars)
        grid_logp.append(mylogp)
    
    #-- convert to arrays
    grid_pars = np.array(grid_pars)
    grid_logp = np.array(grid_logp)
    
    #-- store the info in a feedback dictionary
    feedback = dict(parameters=[],values=[],priors=[])
    
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
                this_param = parset.get_parameter(qual)
                #this_param.set_posterior(trace.ravel())f
                feedback['parameters'].append(this_param)
                feedback['values'].append(grid_pars[:,index])
                feedback['priors'].append(this_param.get_prior(fitter=None)['range'])
    feedback['logp'] = grid_logp
    fitparams['feedback'] = feedback
    return fitparams

#}

#{ Verifiers

def summarize_fit(system,fitparams,savefig=False):
    """
    Summarize a fit.
    
    @param system: the fitted system
    @type system: Body
    @param fitparams: the fit parameters (containing feedback)
    @type fitparams: ParameterSet
    """
    #-- which parameters were fitted?
    fitted_param_labels = [par.get_unique_label() for par in fitparams['feedback']['parameters']]
    #-- walk through all parameterSets until you encounter a parameter that
    #   was used in the fit
    had = []
    indices = []
    paths = []
    units = []
    table = []
    for path,param in system.walk_all(path_as_string=True):
        if not isinstance(param,parameters.Parameter):
            continue
        this_label = param.get_unique_label()
        if this_label in fitted_param_labels and not this_label in had:
            if 'orbit' in path:
                orb_index = path.index('orbit')
                path = path[:orb_index-1]+path[orb_index:]
            index = fitted_param_labels.index(this_label)
            trace = fitparams['feedback']['traces'][index]
            prior = fitparams['feedback']['priors'][index]
            # Raftery-Lewis diagnostics
            req_iters,thin_,burn_in,further_iters,thin = pymc.raftery_lewis(trace,q=0.025,r=0.1,verbose=0)
            thinned_trace = trace[burn_in::thin]
            indtrace = np.arange(len(trace))
            indtrace = indtrace[burn_in::thin]
            if param.has_unit():
                unit = param.get_unit()
            else:
                unit=''
            plt.figure(figsize=(14,8))
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
            
            if prior['distribution']=='uniform':
                plt.axvspan(prior['lower'],prior['upper'],alpha=0.05,color='r')
            elif prior['distribution']=='normal':
                plt.axvspan(prior['mu']-1*prior['sigma'],prior['mu']+1*prior['sigma'],alpha=0.05,color='r')
                plt.axvspan(prior['mu']-3*prior['sigma'],prior['mu']+3*prior['sigma'],alpha=0.05,color='r')
            plt.xlabel("/".join(path)+' [{}]'.format(unit))
            plt.ylabel('Number of occurrences')
            plt.grid()
            plt.subplot(243)
            plt.title("Geweke plot")
            scores = np.array(pymc.geweke(trace)).T
            plt.plot(scores[0],scores[1],'o')
            plt.axhline(-2,color='g',linewidth=2,linestyle='--')
            plt.axhline(+2,color='b',linewidth=2,linestyle='--')
            plt.grid()
            plt.xlabel("Iteration")
            plt.ylabel("Score")
            plt.subplot(244)
            plt.title("Box and whisper plot")
            plt.boxplot(trace)
            plt.ylabel("/".join(path)+' [{}]'.format(unit))
            txtval = '{}=({:.3g}$\pm${:.3g}) {}'.format(path[-1],trace.mean(),trace.std(),unit)
            plt.xlabel(txtval)#,xy=(0.05,0.05),ha='left',va='top',xycoords='axes fraction')
            plt.xticks([])
            plt.grid()
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
    
    #-- print a summary table
    #-- what is the width of the columns?
    col_widths = [9,4,3,3,4]
    for line in table:
        for c,col in enumerate(line):
            col_widths[c] = max(col_widths[c],len(col))
    template = '{{:{:d}s}} = ({{:{:d}s}} +/- {{:{:d}s}}) {{:{:d}s}} {{}}'.format(*[c+1 for c in col_widths])
    header = template.format('Qualifier','Mean','Std','50%','Unit')
    print('='*len(header))
    print(header)
    print('='*len(header))
    for line in table:
        print(template.format(*line))
    print('='*len(header))
    for i,(ipar,iind,ipath) in enumerate(zip(had,indices,paths)):
        for j,(jpar,jind,jpath) in enumerate(zip(had,indices,paths)):
            if j<=i: continue
            plt.figure()
            plt.xlabel("/".join(ipath)+' [{}]'.format(units[i]))
            plt.ylabel("/".join(jpath)+' [{}]'.format(units[j]))
            plt.hexbin(fitparams['feedback']['traces'][iind],
                       fitparams['feedback']['traces'][jind],bins='log',gridsize=50)
            cbar = plt.colorbar()
            cbar.set_label("Number of occurrences")
    print("MCMC path length = {}".format(len(fitparams['feedback']['traces'][iind])))
    return None


def accept_fit(system,fitparams):
    """
    Accept the fit.
    """
    feedback = fitparams['feedback']
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
                if fitparams.context.split(':') in ['pymc','emcee']:
                    this_param.set_value(np.median(feedback['traces'][index]))
                    logger.info("Set {} = {} from MCMC trace".format(qual,this_param.as_string()))
                elif fitparams.context.split(':') in ['lmfit']:
                    this_param.set_value(feedback['values'][index])
                    logger.info("Set {} = {} from {} fit".format(qual,this_param.as_string(),fitparams['method']))


