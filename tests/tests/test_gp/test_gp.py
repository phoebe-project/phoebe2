"""
"""
import phoebe
import numpy as np


def test_gps(verbose=False, plot=False):
    # Make fake data
    b = phoebe.default_binary()

    b.add_dataset('lc', compute_times=phoebe.linspace(0,5.,301))

    #Make a set of fast compute options
    b.add_compute(compute='fast_compute')
    b.set_value_all('ld_mode', value='manual')
    b.set_value('irrad_method', compute='fast_compute', value='none')
    b.set_value_all('distortion_method', compute='fast_compute', value='sphere')
    b.set_value_all('atm', value='ck2004')


    b.run_compute(compute='fast_compute')

    times = b.get_value(qualifier='times', context='model')
    fluxes = b.get_value(qualifier='fluxes', context='model') + np.random.normal(size=times.shape) * 0.07 + 0.2*np.sin(times)
    sigmas = np.ones_like(fluxes) * 0.05

    #Upload the fake data to PHOEBE

    b.add_dataset('lc', dataset='lc01', times=times, fluxes=fluxes, sigmas=sigmas, overwrite=True)
    b.set_value_all('ld_mode', value='manual')

    b.run_compute(model='withoutGPs', compute='fast_compute')

    #Make a model with GPs
    b.add_gaussian_process('celerite2', dataset='lc01', kernel='sho')
    
    if verbose:
        print("initial values of rho, tau and sigma = ", b['rho@gp_celerite201'],b['tau@gp_celerite201'],b['sigma@gp_celerite201'])

    # Compute model in phase space
    b.flip_constraint('compute_phases', solve_for='compute_times')
    b.set_value('compute_phases', phoebe.linspace(-0.5,0.5,101))

    b.run_compute(model='withGPs', compute='fast_compute')

    if plot:
        b.plot(kind='lc', c={'withoutGPs': 'red', 'withGPs': 'green'},
                ls={'withoutGPs': 'dashed', 'withGPs': 'solid'},
                s={'model': 0.03},
                            show=True)
        
    b.add_distribution('rho@gp_celerite201', phoebe.uniform_around(0.5), distribution='init_sample')
    b.add_distribution('tau@gp_celerite201', phoebe.uniform_around(0.5), distribution='init_sample')
    b.add_distribution('sigma@gp_celerite201', phoebe.uniform_around(0.1), distribution='init_sample')
        
    b.add_distribution('rho@gp_celerite201', phoebe.uniform(0.01,2.), distribution='mypriors')
    b.add_distribution('tau@gp_celerite201', phoebe.uniform(0.01,2.), distribution='mypriors')
    b.add_distribution('sigma@gp_celerite201', phoebe.uniform(0.5,1.5), distribution='mypriors')

    b.add_solver('sampler.emcee', solver='mcmc_gps',
              init_from='init_sample',
              priors='mypriors', 
              compute='fast_compute', nwalkers=7, niters=10)
    
    b.run_solver(solver='mcmc_gps', solution = 'mcmc_gps_sol')
    b.adopt_solution('mcmc_gps_sol')

    b.run_compute(model='GPsol', compute='fast_compute')
    if verbose:
        print("fitted values of rho, tau and sigma = ", b['rho@gp_celerite201'],b['tau@gp_celerite201'],b['sigma@gp_celerite201'])
        print ("compute with GPs solved")
    
    if plot:
        b.plot(kind='lc', c={'GPsol': 'red', 'withGPs': 'green'},
                ls={'withGPs': 'dashed', 'GPsol': 'solid'},
                s={'model': 0.03},show=True)

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_gps(verbose=True, plot=True)
