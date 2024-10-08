"""
"""
import phoebe


def test_distance_scaling_alt_backend(verbose=False, plot=False):
    b = phoebe.default_binary()

    b.add_dataset('lc', compute_phases=[0.25])
    for backend in phoebe.list_available_computes():
        if backend == 'phoebe':
            # compute options already exist
            continue
        if backend == 'jktebop':
            # TODO: re-enable this in the test, currently fails on CI
            continue
        b.add_compute(backend)

    b.set_value_all('ld_mode', 'manual')
    b.set_value('distance', 2)
    for compute in b.computes:
        try:
            b.run_compute(compute=compute, model=f'{compute}_at_2')
        except ImportError:
            pass

    if plot:
        b.plot(legend=True, show=True)

    for model in b.models:
        flux = b.get_value(qualifier='fluxes', model=model)[0]
        diff = abs(0.5-flux)
        if verbose:
            print(f"model={model} diff={diff}")
        assert diff < 0.01


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_distance_scaling_alt_backend(verbose=True, plot=True)
