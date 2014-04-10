import phoebe
import os
import numpy as np

def test_access():
    """
    Testing bundle setters and getters
    """
    mybundle = phoebe.Bundle(os.path.join(os.path.dirname(os.path.abspath(__file__)),'defaults.phoebe'))
    
    # test get value
    assert(mybundle.get_value('distance')==10)
    assert(mybundle.get_value('teff@primary')==8350)
    assert(mybundle.get_value('teff->primary')==8350)
    assert(mybundle.get_value('delta@mesh:marching@secondary')==0.0527721121857703257)
    assert(mybundle.get_value('atm@lcdep@secondary')=='kurucz')
    assert(mybundle.get_value('atm@component@secondary')=='kurucz')
    
    # test set value
    mybundle.set_value('teff@primary',9000)
    assert(mybundle.get_value('teff@primary')==9000)
    mybundle.set_value('teff->primary', 9001)
    assert(mybundle.get_value('teff->primary')==9001)
    mybundle.set_value('delta@mesh:marching@secondary', 0.5)
    assert(mybundle.get_value('delta@mesh:marching@secondary')==0.5)
    mybundle.set_value('atm@lcdep@secondary', 'blackbody')
    assert(mybundle.get_value('atm@lcdep@secondary')=='blackbody')
    mybundle.set_value('atm@component@secondary','something')
    assert(mybundle.get_value('atm@component@secondary')=='something')
    
    # add some data and do similar stuff
    mybundle.create_data(category='lc', dataref='mylc', time=np.linspace(0,1,100), flux=np.ones(100))
    mybundle.create_data(category='rv', dataref='myprimrv', objref='primary', time=np.linspace(0,1,100), rv=np.ones(100))
    mybundle.create_data(category='rv', dataref='mysecrv', objref='secondary', time=np.linspace(0.2,0.3,109), rv=-np.ones(109))
    #print mybundle.get_value('time@mylc')
    
    
    
    
    
if __name__ == "__main__":
    test_access()
    