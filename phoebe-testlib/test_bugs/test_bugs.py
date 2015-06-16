import phoebe
import numpy as np


def test_arrsizes():

    """

    test_bugs.test_arrsizes: testing that the size of arrays in rvsyn match rvobs

    """

    eb = phoebe.Bundle()
    time = np.linspace(0, 5*eb['period'], 50)
    eb.rv_fromarrays(time=time, objref='primary', dataref='rv01')
    eb.rv_fromarrays(time=time, objref='secondary', dataref='rv02')
    eb.run_compute()
    print len(eb['rv01@rvsyn@primary']['time']), len(eb['rv01@rvobs@primary']['time']) 
    assert(len(eb['rv01@rvsyn@primary']['time']) == len(eb['rv01@rvobs@primary']['time']))
    
    print len(eb['rv02@rvsyn@secondary']['time']), len(eb['rv02@rvobs@secondary']['time']) 
    assert(len(eb['rv01@rvobs@primary']['time']) == 50)
    assert(len(eb['rv02@rvobs@secondary']['time']) == 50)
    assert(len(eb['rv01@rvsyn@primary']['time']) == 50)
    assert(len(eb['rv02@rvsyn@secondary']['time']) == 50)


    eb['method@rv01'] = 'dynamical'
    eb['method@rv02'] = 'dynamical'

    eb.run_compute()
    print len(eb['rv01@rvsyn@primary']['time']), len(eb['rv01@rvobs@primary']['time']) 
    assert(len(eb['rv01@rvsyn@primary']['time']) == len(eb['rv01@rvobs@primary']['time']))
    
    print len(eb['rv02@rvsyn@secondary']['time']), len(eb['rv02@rvobs@secondary']['time']) 
    assert(len(eb['rv01@rvobs@primary']['time']) == 50)
    assert(len(eb['rv02@rvobs@secondary']['time']) == 50)
    assert(len(eb['rv01@rvsyn@primary']['time']) == 50)
    assert(len(eb['rv02@rvsyn@secondary']['time']) == 50)
    
    
    
    eb = phoebe.Bundle()
    time = np.linspace(0, 5*eb['period'], 50)
    eb.rv_fromarrays(time=time, objref=['primary', 'secondary'], dataref='rv01')
    eb.run_compute()
    print len(eb['rv01@rvsyn@primary']['time']), len(eb['rv01@rvobs@primary']['time']), len(eb['rv01@rvsyn@secondary']['time']),  len(eb['rv01@rvobs@secondary']['time'])
    assert(len(eb['rv01@rvobs@primary']['time']) == 50)
    assert(len(eb['rv01@rvsyn@primary']['time']) == 50)
    assert(len(eb['rv01@rvobs@secondary']['time']) == 50)
    assert(len(eb['rv01@rvsyn@secondary']['time']) == 50)
    
    
    eb['method@rv01@primary'] = 'dynamical'
    eb['method@rv01@secondary'] = 'dynamical'
    eb.run_compute()
    print len(eb['rv01@rvsyn@primary']['time']), len(eb['rv01@rvobs@primary']['time']), len(eb['rv01@rvsyn@secondary']['time']),  len(eb['rv01@rvobs@secondary']['time'])
    assert(len(eb['rv01@rvobs@primary']['time']) == 50)
    assert(len(eb['rv01@rvsyn@primary']['time']) == 50)
    assert(len(eb['rv01@rvobs@secondary']['time']) == 50)
    assert(len(eb['rv01@rvsyn@secondary']['time']) == 50)
    


def test_boostingswitch():
    
    """
    test_bugs.test_boostingswitch: testing that disabling boosting works correctly
    
    """

    b = phoebe.Bundle('KPD1946+4340')
    
    b.lc_fromarrays(phase=np.linspace(0,1,20))
    b['boosting@lc01@lcdep@sdB_primary'] = False
    b['boosting@lc01@lcdep@WD_secondary'] = False
    
    b['boosting_alg@preview'] = 'local'
    
    b.run_compute('preview')
    flux1 = b['flux@lcsyn']
    
    b['boosting_alg@preview'] = 'None'
    b.run_compute('preview')
    
    flux2 = b['flux@lcsyn']
    
    assert(np.allclose(flux1,flux2))
    

if __name__ == "__main__":
    logger = phoebe.get_basic_logger()
    test_arrsizes()
    #~ test_boostingswitch()
