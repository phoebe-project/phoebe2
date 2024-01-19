"""
"""

import phoebe


def test_binary(plot=False):
    b = phoebe.Bundle.default_binary()

    b.add_dataset('lc', times=phoebe.linspace(0, 1, 21))

    b.set_value('Av', 0.2)
    b.flip_constraint('ebv', solve_for='Av')
    b.set_value('ebv', 0.25)

    # enable and implement assertions once extinction tables are available
    # b.run_compute(irrad_method='none')


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_binary(plot=True)
