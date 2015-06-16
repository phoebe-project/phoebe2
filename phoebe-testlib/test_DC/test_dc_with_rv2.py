"""
For testing purposes (if run with nosetests -v), the chain that is run is very
very short, we are just interested to see if everything goes through.

If you run this script as a main script, the number of iterations and walkers
are much larger.
"""
import phoebe
from phoebe.frontend import bundle
import numpy as np
import matplotlib.pyplot as plt

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



    
    

if __name__ == "__main__":
    init_bundle = get_bundle_with_data_and_initial_guess()
    system = init_bundle.get_system()
    
    # Remember the initial values for some parameters, just for comparison report
    # afterwards
    initial_text = []
    for twig in ['ecc', 'per0', 'vgamma', 'sma', 'incl', 'q']:
        initial_text.append(("{:10s} = {:16.8f}".format(twig, init_bundle[twig])))


    # set the fitting method to differential correction
    fitparams = phoebe.ParameterSet('fitting:dc')
    
    # set the stopping criteria type to min_dx
    fitparams.set_value('stopping_criteria_type', 'min_dx')
    
    #set the stop value of min_dx
    fitparams.set_value('stop_value',0.00001)
    
    # set the maximum number of iterations
    fitparams.set_value('max_iters', 30)
    
    # set the derivative_type list  NOTE WELL:  the number of entries MUST
    # match the number of parameters NOT JUST ADJUSTABLE.  If a parameter is not adjustable
    # the entry should be 'none'
    fitparams.set_value('derivative_type',['numerical','numerical','numerical','numerical'])
    
    # Since all the derivatives are set to numerical we set derivative_funcs to "dummy funcs"
    # need to do this for now since c-code checks if they are callable.
    nParams = len(system.get_adjustable_parameters())
    fitparams.set_value('derivative_funcs', [lambda (x, system):x for i in range(nParams)]  )


    params = None
    feedback = phoebe.fitting.run(system, params, fitparams=fitparams, mpi=None, accept=False)

 
    
    print("feedback = ",feedback)
    print("feedback('solution' = ",feedback('solution'))
    
    for i,twig in enumerate(['ecc', 'per0', 'vgamma', 'sma', 'incl', 'q']):
        print(initial_text[i] +' ---> '+"{:10s} = {:16.8f}".format(twig, init_bundle[twig]))

