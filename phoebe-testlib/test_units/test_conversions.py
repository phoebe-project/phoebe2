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
    assert(convert('nm','m/s',455.3,wave=(0.4552,'mum'))==65859.50307564587)
    assert(convert('km/s','AA',65.859503075576129,wave=(4552.,'AA'))==4553.0)
    assert(convert('nm','Ghz',1000.)==299792.4579999999)
    assert(convert('km h-1','nRsol s-1',1.)==0.39930106341859206)
    assert(convert('erg s-1 cm-2 AA-1','SI',1.)==10000000.0)
    assert(np.allclose(convert('erg/s/cm2/AA','Jy',1e-10,wave=(10000.,'angstrom')),333.564095198))
    assert(np.allclose(convert('erg/s/cm2/AA','Jy',1e-10,freq=(constants.cc/1e-6,'hz')),333.564095198))
    assert(np.allclose(convert('erg/s/cm2/AA','Jy',1e-10,freq=(constants.cc,'Mhz')),333.564095198))
    assert(np.allclose(convert('Jy','erg/s/cm2/AA',333.56409519815202,wave=(10000.,'AA')),1e-10))
    assert(np.allclose(convert('Jy','erg/s/cm2/AA',333.56409519815202,freq=(constants.cc,'Mhz')),1e-10))
    assert(np.allclose(convert('W/m2/mum','erg/s/cm2/AA',1e-10,wave=(10000.,'AA')),1.e-11))
    assert(convert('Jy','W/m2/Hz',1.)==1e-26)
    assert(convert('W/m2/Hz','Jy',1.)==1e+26)
    assert(convert('Jy','erg/cm2/s/Hz',1.)==1e-23)
    assert(np.allclose(convert('erg/cm2/s/Hz','Jy',1.),1e+23))
    assert(convert('Jy','erg/s/cm2',1.,wave=(2.,'micron'))==1.49896229e-09)
    assert(np.allclose(convert('erg/s/cm2','Jy',1.,wave=(2.,'micron')),667128190.396))
    # skipped magnitudes and amplitudes
    assert(convert('sr','deg2',1.)==3282.806350011744)
    assert(convert('cy/d','muHz',1.)==11.574074074074074)
    assert(convert('muhz','cy/d',11.574074074074074)==1.0)
    # skipped interferometry
