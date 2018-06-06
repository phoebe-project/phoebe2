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

# check to make sure parameters are the same

    pars = b.filter()
    pars2 = b.filter()

    #checking parameters
    for x in range(len(pars)):
        val1 = pars[x].value
        val2 = pars2[x].value
        if pars[x].qualifier not in ['times', 'fluxes', 'sigmas', 'rvs']:
#            print pars[x].qualifier
#            print pars[x]
            try:
                assert(val1==val2)
            except:
                assert(all(val1==val2))





if __name__ == '__main__':
#    logger= phb2.logger()
    filename = 'default.phoebe'
    #for default binary
    test_reimport()
    #for more complex phoebe1 system
    test_reimport(filename)
