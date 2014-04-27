from phoebe.units.conversions import *
import numpy as np

def test_conversions():
    """
    Unit conversions
    """
    assert(convert('km','cm',1.)==100000.0)
    assert(convert('m/s','km/h',1,0.1)==(3.5999999999999996, 0.36))
    assert(convert('AA','km/s',4553,0.1,wave=(4552.,0.1,'AA'))==(65.85950307557613, 9.314963362464114))
    assert(np.allclose(convert('10mW m-2/nm','erg s-1 cm-2 AA-1',1.),1.0))
    assert(convert('kg','cgs',1.)==1000.0)
    assert(convert('g','SI',1.)==0.001)
    assert(convert('SI','g',1.)==1000.0)
    assert(convert('AA','km/s',4553.,wave=(4552.,'AA'))==65.85950307557613)
    assert(convert('AA','km/s',4553.,wave=(4552.,0.1,'AA'))==(65.85950307557613, 6.587397133195861))