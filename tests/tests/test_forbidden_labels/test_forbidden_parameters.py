"""
"""

import phoebe


def test_forbidden(verbose=False):
    phoebe.devel_on()

    b = phoebe.default_binary(contact_binary=True)
    b.add_dataset('lc')
    b.add_dataset('rv')
    b.add_dataset('mesh')
    b.add_dataset('lp')

    b.add_compute('legacy')
    b.add_compute('photodynam')
    b.add_compute('jktebop')
    b.add_compute('ellc')

    b.add_spot(component='primary')
    b.add_gaussian_process('sklearn', dataset='lc01')
    b.add_gaussian_process('celerite2', dataset='lc01')

    b.add_solver('estimator.lc_periodogram')
    b.add_solver('estimator.rv_periodogram')
    b.add_solver('estimator.lc_geometry')
    b.add_solver('estimator.rv_geometry')
    b.add_solver('estimator.ebai')
    b.add_solver('optimizer.nelder_mead')
    b.add_solver('optimizer.differential_evolution')
    b.add_solver('optimizer.differential_corrections')
    b.add_solver('optimizer.cg')
    b.add_solver('optimizer.powell')
    b.add_solver('sampler.emcee')
    b.add_solver('sampler.dynesty')

    b.add_server('remoteslurm')

    # TODO: include constraint_func?  Shouldn't matter since they're not in twigs
    should_be_forbidden = b.qualifiers + b.contexts + b.kinds + [c.split('@')[0] for c in b.get_parameter(qualifier='columns').choices]

    if verbose:
        for label in should_be_forbidden:
            if label not in phoebe.parameters.parameters._forbidden_labels:
                print(label)

    for label in should_be_forbidden:
        assert label in phoebe.parameters.parameters._forbidden_labels

    phoebe.reset_settings()


if __name__ == '__main__':
    test_forbidden(verbose=True)
