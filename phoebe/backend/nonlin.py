"""
Fitting routines, minimization, optimization.

The top level function you should use is L{run_nonlin}.
"""
import os
import time
import logging
import numpy as np
from matplotlib import pyplot as plt
from phoebe.utils import utils
from phoebe.backend import universe
from phoebe.backend import observatory
from phoebe.wd import wd
from phoebe.parameters import parameters
try:
    import lmfit
except ImportError:
    print("Unable to load nonlinear fitting routines from lmfit: restricted fitting facilities")

    
logger = logging.getLogger("FITTING")

#{ Nonlinear least squares

def run_nonlin(system,params=None,fitparams=None,mpi=None,accept=False):
    """
    Run an MCMC sampler.
    
    The type of MCMC code to use is defined by the context of C{fitparams},
    which is either C{fitting:pymc} or C{fitting:emcee}.
    
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
    utils.add_filehandler(mylogger,flevel='INFO',filename='run_lmfit_{}.log'.format("_".join(time.asctime().split())))
    
    if params is None:
        params = parameters.ParameterSet(context='compute')
        logger.info("No compute parameters given: adding default set")
    if fitparams.context=='fitting:lmfit':
        solver = run_lmfit
    else:
        raise NotImplementedError("Fitting context {} is not understood".format(fitparams.context))
    fitparams = solver(system,params=params,mpi=mpi,fitparams=fitparams)
    #utils.pop_filehandlers(mylogger)
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
    
def run_lmfit(system,params=None,mpi=None,fitparams=None):
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
        return (model-mu)/sigma
    
    #-- do the fit and report on the errors to the screen
    result = lmfit.minimize(model_eval,pars,args=(system,),method=fitparams['method'])
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

#{ Verifiers

def accept_fit(system,fitparams):
    """
    Accept the MCMC fit.
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
                this_param.set_value(feedback['values'][index])
                logger.info("Set {} = {} from {} fit".format(qual,this_param.as_string(),fitparams['method']))

