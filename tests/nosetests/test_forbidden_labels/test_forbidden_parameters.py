"""
"""

import phoebe

def test_forbidden(verbose=False):

    phoebe.devel_on()
    phoebe.check_visible_off()

    b = phoebe.default_binary(contact_binary=True)
    b.add_dataset('lc')
    b.add_dataset('rv')
    b.add_dataset('mesh')
    b.add_dataset('lp')

    b.add_compute('legacy')
    b.add_compute('photodynam')
    b.add_compute('jktebop')
    b.add_compute('ellc')

    # TODO: include constraint_func?  Shouldn't matter since they're not in twigs
    should_be_forbidden = b.qualifiers + b.contexts + b.kinds + [c.split('@')[0] for c in b.get_parameter('columns').choices]

    if verbose:
        for l in should_be_forbidden:
            if l not in phoebe.parameters.parameters._forbidden_labels:
                print(l)

    for l in should_be_forbidden:
        assert(l in phoebe.parameters.parameters._forbidden_labels)

    phoebe.reset_settings()
    return b

if __name__ == '__main__':
    b = test_forbidden(verbose=True)
