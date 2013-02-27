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
                pars.add(name,value=parameter.get_value(),min=prior['lower'],max=prior['upper'],vary=True)
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
    
    result = lmfit.minimize(model_eval,pars,args=(system,),method=fitparams['method'])
    lmfit.report_errors(pars)
    if not result.success:
        print("Warning: nonlinear fit failed!")
    print("Estimating confidence intervals")
    ci = lmfit.conf_interval(result,sigmas=(0.674,0.997))
    lmfit.printfuncs.report_ci(ci)
    feedback = dict(parameters=pars,redchi=result.redchi,success=result.success)
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
            bins = trace.ptp()/(3.5*np.std(trace)/len(trace)**0.333)
            bins_thinned = thinned_trace.ptp()/(3.5*np.std(thinned_trace)/len(thinned_trace)**0.333)
            plt.hist(thinned_trace,bins=bins_thinned,alpha=0.5,normed=True)
            plt.hist(trace,bins=bins,alpha=0.5,normed=True)
            if prior['distribution']=='uniform':
                plt.axvspan(prior['lower'],prior['upper'],alpha=0.05,color='r')
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
                this_param.set_value(np.median(feedback['traces'][index]))
                logger.info("Set {} = {} from MCMC trace".format(qual,this_param.as_string()))

