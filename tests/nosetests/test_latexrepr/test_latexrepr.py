"""
"""

import phoebe

def test_latexrepr(verbose=False):


    b = phoebe.default_binary(contact_binary=True)
    b.add_dataset('lc')
    b.add_dataset('rv')
    b.add_dataset('mesh')
    b.add_dataset('lp')

    b.add_compute('legacy')
    # b.add_compute('photodynam')
    b.add_compute('jktebop')
    b.add_compute('ellc')

    b.add_spot(component='primary')
    b.add_gaussian_process(dataset='lc01')

    for param in b.to_list():
        if verbose:
            print("param: {}".format(param.twig))
        param.latextwig

    b = phoebe.default_star()
    for param in b.to_list():
        if verbose:
            print("param: {}".format(param.twig))
        param.latextwig

    return b

if __name__ == '__main__':
    b = test_latexrepr(verbose=True)
