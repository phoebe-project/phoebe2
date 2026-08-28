import numpy as np
import phoebe


def test_parse_solver_times():
    b = phoebe.default_binary()
    b.add_dataset('lc', times=phoebe.linspace(0,1,301), compute_phases=phoebe.linspace(0,1,101))
    b.set_value('mask_enabled', True)
    b.set_value('mask_phases', [(-0.1, 0.1), (0.45,0.55)])

    b.add_solver('optimizer.nelder_mead')
    b.set_value('solver_times', 'times')
    b.parse_solver_times()


def test_calculate_lnlikelihood_with_mesh_dataset_present():
    b = phoebe.default_binary()
    times = phoebe.linspace(0, 1, 5)
    synthetic_fluxes = np.ones(5)
    synthetic_sigmas = np.full(5, 1e-3)

    b.add_dataset(
        'lc',
        dataset='lc01',
        times=times,
        fluxes=synthetic_fluxes,
        sigmas=synthetic_sigmas,
        passband='Johnson:V',
    )
    b.add_dataset('mesh', dataset='mesh01', compute_times=[0.0])

    b.add_compute(compute='fast_compute')
    b.set_value('irrad_method', compute='fast_compute', value='none')
    b.set_value_all('distortion_method', compute='fast_compute', value='sphere')
    b.set_value_all('ntriangles', compute='fast_compute', value=100)

    b.run_compute(compute='fast_compute', model='fit_model')
    lnlike = b.calculate_lnlikelihood(model='fit_model')

    assert np.isfinite(lnlike)


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_parse_solver_times()