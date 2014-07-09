from phoebe.wd import wd
from phoebe.parameters import parameters
import numpy as np
import os

basedir = os.path.dirname(os.path.abspath(__file__))

def test_wd():
    """
    Test Python interface to WD
    """
    bps = parameters.ParameterSet(context='root', frame='wd')
    out = wd.lc(bps)[0]
    
    outdata = np.column_stack([out[key] for key in out.dtype.names])
    #np.savetxt('output.test', outdata)
    test = np.loadtxt(os.path.join(basedir,'output.test'))
    
    assert(np.allclose(test, outdata))
     
    
    
if __name__ == "__main__":
    test_wd()