import phoebe
import numpy as np


def test_nm_continue():
    # modified from https://phoebe-project.org/docs/2.4/tutorials/nelder_mead
    b = phoebe.default_binary()
    b.set_value(qualifier='ecc', value=0.2)
    b.set_value(qualifier='per0', value=25)
    b.set_value(qualifier='teff', component='primary', value=7000)
    b.set_value(qualifier='teff', component='secondary', value=6000)
    b.set_value(qualifier='sma', component='binary', value=7)
    b.set_value(qualifier='incl', component='binary', value=80)
    b.set_value(qualifier='q', value=0.3)
    b.set_value(qualifier='t0_supconj', value=0.1)
    b.set_value(qualifier='requiv', component='primary', value=2.0)
    b.set_value(qualifier='vgamma', value=80)

    lctimes = phoebe.linspace(0, 10, 15)
    rvtimes = phoebe.linspace(0, 10, 15)
    b.add_dataset('lc', compute_times=lctimes)
    b.add_dataset('rv', compute_times=rvtimes)

    b.run_compute(irrad_method='none')

    fluxes = b.get_value('fluxes@model') + np.random.normal(size=lctimes.shape) * 0.01
    fsigmas = np.ones_like(lctimes) * 0.02

    rvsA = b.get_value('rvs@primary@model') + np.random.normal(size=rvtimes.shape) * 10
    rvsB = b.get_value('rvs@secondary@model') + np.random.normal(size=rvtimes.shape) * 10
    rvsigmas = np.ones_like(rvtimes) * 20


    b = phoebe.default_binary()

    b.add_dataset('lc',
                  compute_phases=phoebe.linspace(0,1,5),
                  times=lctimes,
                  fluxes=fluxes,
                  sigmas=fsigmas,
                  dataset='lc01')

    b.add_dataset('rv',
                  compute_phases=phoebe.linspace(0,1,5),
                  times=rvtimes,
                  rvs={'primary': rvsA, 'secondary': rvsB},
                  sigmas=rvsigmas,
                  dataset='rv01')

    b.set_value('irrad_method', 'none')

    b.set_value(qualifier='sma', component='binary', value=7+0.5)
    b.set_value(qualifier='incl', component='binary', value=80+10)
    b.set_value(qualifier='q', value=0.3+0.1)
    b.set_value(qualifier='t0_supconj', value=0.1)
    b.set_value(qualifier='requiv', component='primary', value=2.0-0.3)

    b.add_solver('optimizer.nelder_mead', solver='nm_solver',
                 fit_parameters=['teff', 'requiv'], maxiter=3)

    b.run_solver(solver='nm_solver', solution='nm_solution')

    b.add_solver('optimizer.nelder_mead', solver='nm_solver_contd',
                 continue_from='nm_solution', maxiter=3)
    b.run_solver(solver='nm_solver_contd', solution='nm_solution_contd')


if __name__ == '__main__':
    test_nm_continue()