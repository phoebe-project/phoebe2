import nose
import phoebe
import numpy as np
import os





def check_parse(myfile, type='lc'):
    """
    Compare the contents and shape of data read in by Phoebe and by Numpy.
    
    Bottom line: we need to have as many columns and lines, and, assuming that
    the first column is time, all numbers need to be the same after read in by
    Numpy and by Phoebe.
    """
    # Read via Phoebe
    print("filename: {}, type: {}".format(myfile,type))
    if type == 'lc':
        obs, pbdep = phoebe.parse_lc(myfile)
    elif type == 'rv':
        obs, pbdep = phoebe.parse_rv(myfile)
        
    # Read via Numpy
    data = np.loadtxt(myfile)
    
    # We need to have as many lines:
    assert(data.shape[0] == obs[0].shape[0])
    
    # We need to have as many columns:
    assert(data.shape[1] == len(obs[0]['columns']))
    
    # First column is most probably time column (strictly speaking this is not
    # necessary for the parser, but we assume it here in the tests)
    assert(np.all(data[:,0] == obs[0]['time']))
    
    
    
    
def test_parse_lc():
    
    lc_files = ['asas.V', 'davidge.B', 'davidge.U', 'davidge.V', 'pritchard.b',
                'pritchard.Ic', 'pritchard.u', 'pritchard.v', 'pritchard.V',
                'pritchard.y']
    lc_files = [os.path.join('../phoebe-testlib/hv2241', lc_file) for lc_file in lc_files]
    
    for lc_file in lc_files:
        check_parse(lc_file, type='lc')
    
def test_parse_rv():
    
    rv_files = ['hv2241.final.rv1', 'hv2241.final.rv2']
    rv_files = [os.path.join('../phoebe-testlib/hv2241', rv_file) for rv_file in rv_files]
    
    for rv_file in rv_files:
        check_parse(rv_file, type='rv')
    
    
