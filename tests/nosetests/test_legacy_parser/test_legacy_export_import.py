import phoebe as phb2
import numpy as np

def test_reimport(filename=None):

#import data

    if filename:
        b = phb2.from_legacy(filename)
    else:
        b = phb2.default_binary()
        b.add_compute(kind='legacy')

    b.export_legacy('test.legacy')
    b2 = phb2.from_legacy('test.legacy')

# compare data

# check to see if datasets are attached and the right number
    Nlcs = len(b.get_dataset(kind='lc').datasets)
    Nlcs2 = len(b2.get_dataset(kind='lc').datasets)
    Nrvs = len(b.get_dataset(kind='rv').datasets)
    Nrvs2 = len(b2.get_dataset(kind='rv').datasets)

    # must be equal
    assert(Nlcs==Nlcs2)
    assert(2*Nrvs==Nrvs2)

# check to make sure parameters are the same

    pars = b.filter()
    pars2 = b.filter()

    #checking parameters
    for x in range(len(pars)):
        val1 = pars[x].value
        val2 = pars2[x].value
        if pars[x].qualifier not in ['times', 'fluxes', 'sigmas', 'rvs']:
            
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
