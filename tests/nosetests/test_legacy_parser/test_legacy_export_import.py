import phoebe as phb2
import numpy as np

phb2.devel_on()

def test_reimport(filename=None):

#import data

    if filename:
        b = phb2.from_legacy(filename)
    else:
        b = phb2.default_binary()

    b.export_legacy('test.legacy')
    b2 = phb2.from_legacy('test.legacy')

# compare data

# check to see if datasets are attached and the right number

    N = len(b.datasets)
    N2 = len(b2.datasets)
    # must be equal
    assert(N==N2)

    if N > 1:
        b.run_compute(kind='phoebe')
        b2.run_compute(kind='phoebe')

        lcs = b.get_dataset(kind='lc').datasets
        lcs = lcs[::-1]
        rvs = b.get_dataset(kind='rv').datasets
        rvs = rvs[::-1]

        for x in range(lcs):
            lc = b.filter('fluxes', context='model', dataset=lcs[x]).get_value()
            lc2 = b2.filter('fluxes', context='model', dataset=lcs[x]).get_value()
            print("checking lc"+str(lcs[x]))
            assert(np.allclose(lc, lc2, atol=1e-5))

            for x in range(rvs):
                rv = b.filter('rvs', component=comp_name, context='model').get_value()
                rv2 = b2.filter('rvs', component=comp_name, context='model').get_value()
                print("checking rv"+str(rvs[x]))
                assert(np.allclose(rv, rv2, atol=1e-5))

if __name__ == '__main__':
#    logger= phb2.logger()
    filename = 'default.phoebe'
    #for default binary
    test_reimport()
    #for more complex phoebe1 system
    test_reimport(filename)
