import phoebe
import numpy as np


def test_arrsizes():

    """

    testing that the size of arrays in rvsyn match rvobs

    """

    eb = phoebe.Bundle()
    time = np.linspace(0,5.*eb['period'], 50)
    eb.rv_fromarrays(time = time, objref = 'primary', dataref='rv01')
    eb.run_compute()
    print len(eb['rv01@rvsyn@primary']['time']), len(eb['rv01@rvobs@primary']['time']) 
    assert(len(eb['rv01@rvsyn@primary']['time']) == len(eb['rv01@rvobs@primary']['time']))
    assert(len(eb['rv01@rvsyn@secondary']['time']) == len(eb['rv01@rvobs@secondary']['time']))


if __name__ == "__main__":
    logger = phoebe.get_basic_logger()
    test_arrsizes()
