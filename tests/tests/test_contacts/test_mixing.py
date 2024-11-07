import phoebe
import numpy as np


def test_mixing(verbose=False):
    if verbose:
        print('phoebe.default_contact_binary()')
    b = phoebe.default_contact_binary()
    b.add_dataset('mesh', compute_times=[0])
    b.set_value('teff@secondary', 3000)  # so we have different temperatures
    b.set_value('mixing_enabled', True)
    b.set_value('mixing_method', 'perfect')  # since other mixing methods are arbitrarily parametrized, test this case only
    b.set_value('columns', '*')  # to populate 'teffs'

    assert b.run_checks().passed

    b.run_compute()

    assert (res := abs(1 - np.mean(b['teffs@secondary'].value) / np.mean(b['teffs@primary'].value))) <= 0.01
    if verbose:
        print('relative teff difference', res)


if __name__ == '__main__':
    test_mixing(verbose=True)
