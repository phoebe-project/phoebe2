import phoebe
import os
import numpy as np
import nose.tools

def test_access():
    """
    Testing bundle setters and getters
    """
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),'defaults.phoebe')
    mybundle = phoebe.Bundle(filename)
    # set a new parameterSet
    mybundle.attach_ps('Detached_1', phoebe.PS('reddening:interstellar'))
    
    # test get value
    assert(mybundle.get_value('vgamma@position')==-12.5)
    assert(mybundle.get_value('teff@primary')==8350)
    assert(mybundle.get_value('label@primary')=='primary')
    assert(mybundle.get_value('delta@mesh:marching@secondary')==0.0527721121857703257)
    assert(mybundle.get_value('atm@lcdep@secondary')=='kurucz')
    assert(mybundle.get_value('atm@component@secondary')=='kurucz')
    assert(mybundle.get_value('passband@reddening')=='JOHNSON.V')
    
    # test set value
    mybundle.set_value('teff@primary',9000)
    assert(mybundle.get_value('teff@primary')==9000)
    mybundle.set_value('teff@primary', 9001)
    assert(mybundle.get_value('teff@primary')==9001)
    mybundle.set_value('delta@mesh:marching@secondary', 0.5)
    assert(mybundle.get_value('delta@mesh:marching@secondary')==0.5)
    mybundle.set_value('atm@lcdep@secondary', 'blackbody')
    assert(mybundle.get_value('atm@lcdep@secondary')=='blackbody')
    mybundle.set_value('atm@component@secondary','kurucz')
    assert(mybundle.get_value('atm@component@secondary')=='kurucz')
    
    
    
    # add some data and do similar stuff
    mybundle.create_data(category='lc', dataref='mylc', time=np.linspace(0,1,100), flux=np.ones(100))
    mybundle.create_data(category='rv', dataref='myprimrv', objref='primary', time=np.linspace(0,1,100), rv=np.ones(100))
    mybundle.create_data(category='rv', dataref='mysecrv', objref='secondary', time=np.linspace(0.2,0.3,109), rv=-np.ones(109))
    #print mybundle.get_value('time@mylc')
    
@nose.tools.raises(ValueError)    
def test_error():
    """
    Testing error raise 
    """
    mybundle = phoebe.Bundle(os.path.join(os.path.dirname(os.path.abspath(__file__)),'defaults.phoebe'))
    mybundle.get_value('teff')
    
    
if __name__ == "__main__":
    test_access()
    test_error()
    
