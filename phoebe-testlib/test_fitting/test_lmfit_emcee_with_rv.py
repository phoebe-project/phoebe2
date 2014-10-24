"""
For testing purposes (if run with nosetests -v), the chain that is run is very
very short, we are just interested to see if everything goes through.

If you run this script as a main script, the number of iterations and walkers
are much larger.
"""
from phoebe.frontend import bundle
from phoebe.backend import fitting
from phoebe.parameters import parameters
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1111)

fitting.fit_logger_level = 'CRITICAL'

def test_fitting():
    """
    Test lmfit and emcee + resampling from posteriors on RV
    """
    run_as_main = __name__ == "__main__"

    # Initiate a Bundle, change some parameters and compute an rv curve
    mybundle = bundle.Bundle('binary')
    mybundle['ecc'] = 0.34
    mybundle['per0'] = 53.0
    mybundle['vgamma'] = -10.0
    mybundle['sma'] = 10.0
    mybundle['incl'] = 67.0
    mybundle['q'] = 0.66

    # Info on the mass:
    #print(phoebe.compute_mass_from(mybundle))

    # Generate a radial velocity curve
    time = np.sort(np.random.uniform(low=0, high=10*mybundle['period'], size=50))
    pos, velo, btime, ptime = mybundle.get_orbit('primary', time=time)
    sigma = 5*np.ones(len(time))
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
    mybundle.set_adjust('asini')
    mybundle.set_adjust('incl')
    mybundle.set_adjust('q')

    # Set iniitial values
    mybundle['ecc'] = 0.0
    mybundle['per0'] = 90.0
    mybundle['vgamma'] = 0.0
    mybundle['incl'] = 80.0
    mybundle['q'] = 0.5

    # Specify two fitting routines
    iters = 100 if run_as_main else 20
    walkers = 30 if run_as_main else 10
    mybundle.add_fitting(context='fitting:emcee', iters=iters, walkers=walkers,
                        init_from='previous_run', label='mcmc',
                        computelabel='preview', incremental=True)
    mybundle.add_fitting(context='fitting:lmfit', label='leastsq', method='leastsq',
                        computelabel='preview')

    if not run_as_main:
        # Run least square but do not accept results (just a test run)
        mybundle.run_fitting(label='leastsq', accept_feedback=False)

    # Run mcmc fitting first time
    
    mybundle.add_mpi(label='np6', np=6)
    #~ mpi = None if not run_as_main else parameters.ParameterSet('mpi', np=6)
    mpilabel = 'np6' if run_as_main else 'None'
    mybundle.run_fitting(label='mcmc', mpilabel=mpilabel)
    
    if not run_as_main:
        # Run fitting after resampling from posterior, but with a cutoff in lnproblim
        mybundle['mcmc@feedback'].modify_chain(lnproblim=-60)#-35)
        mybundle.accept_feedback('mcmc')
        mybundle['init_from@mcmc@fitting'] = 'posterior'
        # Continue fitter
        mybundle['init_from@mcmc@fitting'] = 'previous_run'
        feedback = mybundle.run_fitting(label='mcmc', mpilabel=mpilabel)    
    else:
        # Resample from posteriors with a lnproblim cutoff
        mybundle['mcmc@feedback'].modify_chain(lnproblim=-35)
        mybundle.accept_feedback('mcmc')
        mybundle['init_from@mcmc@fitting'] = 'posterior'
        feedback = mybundle.run_fitting(label='mcmc', mpilabel=mpilabel)
        # Continue fitter (see what the acceptance fraction does)
        mybundle['init_from@mcmc@fitting'] = 'previous_run'
        feedback = mybundle.run_fitting(label='mcmc', mpilabel=mpilabel)
        
    

    ## And plot results
    if run_as_main:
        plt.figure()
        mybundle['mcmc@feedback'].plot_logp()
        mybundle['mcmc@feedback'].plot_summary()
        plt.figure()
        mybundle['mcmc@feedback'].plot_acceptance_fraction()
        plt.figure()
        mybundle['mcmc@feedback'].plot_history()
        
        
        plt.show()
    
    # If we get here, that's fine enough for us
    assert(True)

if __name__ == "__main__":
    fitting.fit_logger_level = 'WARNING'
    test_fitting()
