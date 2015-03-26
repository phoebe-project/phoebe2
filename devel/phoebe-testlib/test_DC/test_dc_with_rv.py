"""
For testing purposes (if run with nosetests -v), the chain that is run is very
very short, we are just interested to see if everything goes through.

If you run this script as a main script, the number of iterations and walkers
are much larger.
"""
from phoebe.frontend import bundle
import numpy as np
import matplotlib.pyplot as plt
from phoebe.algorithms import _DiffCorr

np.random.seed(1111)

def get_bundle_with_data_and_initial_guess():
    """
    Test lmfit and emcee + resampling from posteriors on RV
    """
    run_as_main = __name__ == "__main__"

    # Initiate a Bundle, change some parameters and compute an rv curve
    mybundle = bundle.Bundle('binary')
    #mybundle['ecc'] = 0.34
    #mybundle['per0'] = 53.0
    #mybundle['vgamma'] = -10.0
    #mybundle['sma'] = 10.0
    #mybundle['incl'] = 67.0
    #mybundle['q'] = 0.66

    mybundle['ecc'] = 0.32
    mybundle['per0'] = 53.5
    mybundle['vgamma'] = -10.5
    mybundle['sma'] = 10.5
    mybundle['incl'] = 67.66
    mybundle['q'] = 0.68



    # Generate a radial velocity curve
    time = np.sort(np.random.uniform(low=0, high=mybundle['period'], size=50))
    pos, velo, btime, ptime = mybundle.get_orbit('primary', time=time)
    sigma = 0.005*np.ones(len(time))
    noise = np.random.normal(scale=sigma)

    # Add the RV curve as data
    mybundle.rv_fromarrays('primary', time=time, rv=velo[2]+noise, sigma=sigma, method='dynamical')

    # Add custom parameters
    mybundle.add_parameter('mass1@orbit')
    mybundle.add_parameter('asini@orbit', replaces='sma')

    # Define priors
    mybundle.set_prior('ecc', distribution='uniform', lower=0, upper=1)
    mybundle.set_prior('per0', distribution='uniform', lower=0, upper=360)
    mybundle.set_prior('vgamma', distribution='uniform', lower=-30, upper=10)
    mybundle.set_prior('incl', distribution='uniform', lower=0, upper=90)
    mybundle.set_prior('q', distribution='uniform', lower=0.5, upper=1)
    mybundle.set_prior('sma', distribution='uniform', lower=8, upper=12)
    mybundle.set_prior('mass1', distribution='uniform', lower=0.8, upper=1.2)
    mybundle.set_prior('asini@orbit', distribution='uniform', lower=0, upper=15)

    # Mark adjustables
    mybundle.set_adjust('ecc')
    mybundle.set_adjust('per0')
    mybundle.set_adjust('vgamma')
    #mybundle.set_adjust('asini')
    #mybundle.set_adjust('incl')
    mybundle.set_adjust('q')
    
    # Set initial values (closeby)
    mybundle['ecc'] = 0.33
    mybundle['per0'] = 54.0
    mybundle['vgamma'] = -11.0
    #mybundle['incl'] = 68.0
    mybundle['q'] = 0.67
    
    return mybundle


def run_dc(system, params=None, mpi=None, fitparams=None):
    
    #cheap way of faking enum
    #derivative type
    NUMERICAL = 0
    ANALYTICAL = 1
    NONE = 2

    # stopping criteria type
    MIN_DX = 0
    MIN_DELTA_DX = 1
    MIN_CHI2 = 2
    MIN_DELTA_CHI2 = 3
    
    # For DC, the initial values need to make sense
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
    ppars = [] # Phoebe parameters
    initial_guess = []
    bounds = []
    derivative_type = []
    
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
                myid = parameter.get_unique_label().replace('-','_')
                if myid in ids:
                    continue
                
                #-- else, add the name to the list of pnames. Ask for the
                #   prior of this object, we can use this as bounds
                prior = parameter.get_prior()
                minimum, maximum = prior.get_limits()
                
                # add the bounds
                bounds.append((minimum, maximum))
                
                #-- and add the id and values
                ids.append(myid)
                initial_guess.append(parameter.get_value())
                ppars.append(parameter)
                derivative_type.append(NUMERICAL)
    
    # Get the names of the qualifiers for reporting reasons
    qualifiers = [ippar.get_qualifier() for ippar in ppars]
    
    # Current state: we know all the parameters that need to be fitted (and
    # their values and bounds), and we have unique string labels to refer to them.
    
    # Next up: define a function to evaluate the model given a set of
    # parameter values. You'll see that a lot of the code is similar as the
    # one above; but instead of just keeping track of the parameters, we
    # change the value of each parameter. This function will be called by the
    # fitting algorithm itself.
    
    # Remember where we've been
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
                    parset[qual] = pars[ids.index(myid)]
                    had.append(myid)
                    
        # Current state: the system has been updated with the newly proposed
        # values.
        
        # Next up: compute the model given the current (new) parameter values
        system.reset()
        system.clear_synthetic()
        system.compute(params=params, mpi=mpi)
        
        # Retrieve the model
        data, sigma, model = system.get_model()
        
        # Report parameters and chi2
        report_values = ['{}={:16.8f}'.format(qual, val) for qual, val in zip(qualifiers, pars)]
        report_chi2 = np.mean((data-model)**2/sigma**2)
        print(", ".join(report_values) + ': chi2={}'.format(report_chi2))
        
        traces.append(pars)
        redchis.append(report_chi2)
        
        # Return the model
        return model
    
    # Current state: we now have the system to fit, we have an initial guess
    # of the parameters, and we have a function to evaluate the system given new
    # parameters. Next up: initialise some extra arguments for DC    
    initial_guess = np.array(initial_guess)
    derivative_type = np.array(derivative_type, dtype=np.int32)
    print("initial guess = ",initial_guess)
    
    # Get the model, just to know how many datapoints we have. Derive the number
    # of datapoints, derivative functions etc...
    data, sigma, model = system.get_model()
    nDataPoints = len(data)
    nParams = len(initial_guess)
    
    # Fake some time points -- do we really need to get the times from the
    # obs? Probably we do to compute the derivatives? <pieterdegroote>
    time = np.arange(nDataPoints)
    derivative_funcs = [lambda (x, system):x for i in range(nParams)] # dummy derivative funcs

    stoppingCriteriaType = MIN_DX
    #stopValue = 0.0001
    stopValue = 0.00001
    maxIterations = 30



    # Current state: we know which parameters to fit, and we have a function
    # that returns a fit statistic given a proposed set of parameter values.
    
    # Next up: run the fitting algorithm!
    print("about to call DC")
    solution = _DiffCorr.DiffCorr(nParams,  \
				nDataPoints, \
				maxIterations, \
				stoppingCriteriaType, \
				stopValue, \
				data, \
                                initial_guess, derivative_type,
                                derivative_funcs, model_eval, system)

    print solution
    
    # plot history
    traces = np.array(traces).T
    print traces.shape
    for trace, qual in zip(traces, qualifiers):
        plt.figure()
        plt.subplot(211)
        plt.plot(trace, redchis)
        plt.xlabel(qual)
        plt.ylabel('chi2')
        plt.subplot(212)
        plt.plot(trace)
        plt.xlabel("Iteration")
        plt.ylabel(qual)
    plt.show()
    
    

if __name__ == "__main__":
    init_bundle = get_bundle_with_data_and_initial_guess()
    system = init_bundle.get_system()
    
    # Remember the initial values for some parameters, just for comparison report
    # afterwards
    initial_text = []
    for twig in ['ecc', 'per0', 'vgamma', 'sma', 'incl', 'q']:
        initial_text.append(("{:10s} = {:16.8f}".format(twig, init_bundle[twig])))
    
    
    
    run_dc(system)
    
    
    
    
    for i,twig in enumerate(['ecc', 'per0', 'vgamma', 'sma', 'incl', 'q']):
        print(initial_text[i] +' ---> '+"{:10s} = {:16.8f}".format(twig, init_bundle[twig]))

