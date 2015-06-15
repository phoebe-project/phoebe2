import phoebe
import numpy as np
import pylab as pl

def test_oversampling():
    """
    Oversampling: light curve
    """
    bundle1 = phoebe.Bundle('KPD1946+4340')
    bundle1.data_fromarrays(time=[0],exptime=[60.],samprate=[5.],passband='KEPLER.V', dataref='keplc')
    bundle1.run_compute(boosting_alg='simple')
    b1dat = bundle1.get_syn('keplc@KPD1946+4340') # doesn't work
    #b1dat = bundle1.get_system().get_synthetic()
    #print bundle1
    assert(np.allclose(b1dat['time'][0],0.))

def test_boosting():
    """
    Boosting: amplitude match
    """
    phases = [0.75]
    bundle1 = phoebe.Bundle('KPD1946+4340')
    bundle1.data_fromarrays(phase=phases,passband='KEPLER.V', dataref='keplc')
    bundle1.run_compute(boosting_alg='simple')

    bundle2 = phoebe.Bundle('KPD1946+4340')

    bundle2.data_fromarrays(phase=phases,passband='KEPLER.V', dataref='keplc')
    bundle2.run_compute(boosting_alg='none')
    b1dat = bundle1.get_syn('keplc@KPD1946+4340') # doesn't work
    b2dat = bundle2.get_syn('keplc@KPD1946+4340') # doesn't work
    #b1dat = bundle1.get_system().get_synthetic()
    #b2dat = bundle2.get_system().get_synthetic()
        
    assert(b1dat['flux'][0]/b2dat['flux'][0]>1.0007 and b1dat['flux'][0]/b2dat['flux'][0]<1.0008)


if __name__ == "__main__":
    logger = phoebe.get_basic_logger()
    test_oversampling()
    test_boosting()
