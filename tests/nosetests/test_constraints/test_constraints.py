"""
"""

import phoebe
from nose.tools import assert_raises

def test_esinw_ecosw(verbose=False):
    if verbose: print("b=phoebe.default_binary()")
    b = phoebe.default_binary()

    if verbose: print("b.set_value('ecc', 0.5)")
    b.set_value('ecc', 0.5)

    if verbose: print(b.filter(qualifier=['ecosw', 'esinw', 'ecc', 'per0']))

    assert(b.get_value('ecosw', context='component')==0.5)
    assert(b.get_value('esinw', context='component')==0.0)
    assert(b.run_checks()[0])

    if verbose: print("b.flip_constraint('esinw', solve_for='ecc')")
    b.flip_constraint('esinw', solve_for='ecc')

    if verbose: print(b.filter(qualifier=['ecosw', 'esinw', 'ecc', 'per0']))

    assert(b.run_checks()[0]==False)
    assert_raises(ValueError, b.run_compute)

    if verbose: print("b.set_value('per0', 10)")
    b.set_value('per0', 10)

    if verbose: print(b.filter(qualifier=['ecosw', 'esinw', 'ecc', 'per0']))

    assert(b.run_checks()[0])

    assert(b.get_value('ecc', context='component')==0)

    if verbose: print(b.set_value('per0', 0.0))
    b.set_value('per0', 0.0)

    if verbose: print(b.filter(qualifier=['ecosw', 'esinw', 'ecc', 'per0']))

    assert(b.run_checks()[0]==False)

    if verbose: print("b.flip_constraint('ecosw', solve_for='per0')")
    b.flip_constraint('ecosw', solve_for='per0')

    if verbose: print(b.filter(qualifier=['ecosw', 'esinw', 'ecc', 'per0']))

    assert(b.run_checks()[0])
    assert(b.get_value('ecc', context='component')==0.0)
    assert(b.get_value('per0', context='component')==0.0)

    if verbose: print("b.set_value('ecosw', 0.5)")
    b.set_value('ecosw', 0.5)

    assert(b.run_checks())

    if verbose: print(b.filter(qualifier=['ecosw', 'esinw', 'ecc', 'per0']))

    assert(b.get_value('per0', context='component')==0.0)
    assert(b.get_value('ecc', context='component')==0.5)


    return b


if __name__ == '__main__':
    logger = phoebe.logger(clevel='WARNING')

    b = test_esinw_ecosw(verbose=True)
