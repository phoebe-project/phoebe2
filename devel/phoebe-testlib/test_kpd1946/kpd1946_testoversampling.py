import phoebe
import numpy as np
import pylab as pl

def test_oversampling():
    bundle1 = phoebe.Bundle('KPD1946+4340')
    bundle1.create_data(time=[0],exptime=[60.],samprate=[5.],passband='KEPLER.V', dataref='keplc')
    bundle1.run_compute(beaming_alg='simple')

    b1dat = bundle1.get_syn('keplc', objref='KPD1946+4340')
    print b1dat
    assert(np.allclose(b1dat['time'][0],0.))

def test_beaming():
    bundle1 = phoebe.Bundle('KPD1946+4340')
    bundle1.create_data(phase=[0.25],passband='KEPLER.V', dataref='keplc')
    bundle1.run_compute(beaming_alg='simple')

    bundle2 = phoebe.Bundle('KPD1946+4340')

    bundle2.create_data(phase=[0.25],passband='KEPLER.V', dataref='keplc')
    bundle2.run_compute(beaming_alg='none')
    b1dat = bundle1.get_syn('keplc', objref='KPD1946+4340')
    b2dat = bundle2.get_syn('keplc', objref='KPD1946+4340')
    assert(b1dat['flux'][0]/b2dat['flux'][0]>1.0004 and b1dat['flux'][0]/b2dat['flux'][0]<1.0006)


if __name__ == "__main__":
    logger = phoebe.get_basic_logger()
    test_oversampling()
    #test_beaming()
