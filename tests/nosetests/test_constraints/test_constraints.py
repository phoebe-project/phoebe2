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
    assert(b.run_checks().passed)

    if verbose: print("b.flip_constraint('esinw', solve_for='ecc')")
    b.flip_constraint('esinw', solve_for='ecc')

    if verbose: print(b.filter(qualifier=['ecosw', 'esinw', 'ecc', 'per0']))

    assert(b.run_checks().passed==False)
    assert_raises(ValueError, b.run_compute)

    if verbose: print("b.set_value('per0', 10)")
    b.set_value('per0', 10)

    if verbose: print(b.filter(qualifier=['ecosw', 'esinw', 'ecc', 'per0']))

    assert(b.run_checks().passed)

    assert(b.get_value('ecc', context='component')==0)

    if verbose: print(b.set_value('per0', 0.0))
    b.set_value('per0', 0.0)

    if verbose: print(b.filter(qualifier=['ecosw', 'esinw', 'ecc', 'per0']))

    assert(b.run_checks().passed==False)

    if verbose: print("b.flip_constraint('ecosw', solve_for='per0')")
    b.flip_constraint('ecosw', solve_for='per0')

    if verbose: print(b.filter(qualifier=['ecosw', 'esinw', 'ecc', 'per0']))

    assert(b.run_checks().passed)
    assert(b.get_value('ecc', context='component')==0.0)
    assert(b.get_value('per0', context='component')==0.0)

    if verbose: print("b.set_value('ecosw', 0.5)")
    b.set_value('ecosw', 0.5)

    assert(b.run_checks().passed)

    if verbose: print(b.filter(qualifier=['ecosw', 'esinw', 'ecc', 'per0']))

    assert(b.get_value('per0', context='component')==0.0)
    assert(b.get_value('ecc', context='component')==0.5)


    return b

def test_pot_filloutfactor(verbose=False):
    if verbose: print("b=phoebe.default_binary(contact_binary=True)")
    b = phoebe.default_binary(contact_binary=True)

    b.flip_constraint('pot', solve_for='requiv@primary')
    b.flip_constraint('fillout_factor', solve_for='pot')

    b.set_value('fillout_factor', 0.45)
    b.set_value('q', 0.25)

    assert(b.run_checks().passed)

if __name__ == '__main__':
    logger = phoebe.logger(clevel='WARNING')

    b = test_esinw_ecosw(verbose=True)

    b = test_pot_filloutfactor(verbose=True)
