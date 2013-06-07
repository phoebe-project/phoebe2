"""
Unit tests for data parsers.
"""

import nose
import phoebe
from phoebe import kelly
import numpy as np
import os





def check_parse_with_numpy(myfile, type='lc', **kwargs):
    """
    Compare the contents and shape of data read in by Phoebe and by Numpy.
    
    Bottom line: we need to have as many columns and lines, and, assuming that
    the first column is time, all numbers need to be the same after read in by
    Numpy and by Phoebe.
    """
    # Read via Phoebe
    print("filename: {}, type: {}".format(myfile,type))
    if type == 'lc':
        obs, pbdep = phoebe.parse_lc(myfile, **kwargs)
    elif type == 'rv':
        obs, pbdep = phoebe.parse_rv(myfile, **kwargs)
        
    # Read via Numpy
    data = np.loadtxt(myfile)
    
    # We need to have as many lines:
    assert(data.shape[0] == obs[0].shape[0])
    
    # We can't really test the number of columns, as the sigma column is
    # added if it's not there
    
    # Test if the columns contain the same data:
    columns = obs[0]['columns']
    for col in range(min(len(columns),data.shape[1])):
        # Sigma column could have been added, so we don't test for it
        if columns[col] == 'sigma':
            continue
        assert(np.all(data[:,col] == obs[0][columns[col]]))
    
    
    
    
def test_parse_lc_hv2241():
    
    lc_files = ['asas.V', 'davidge.B', 'davidge.U', 'davidge.V', 'pritchard.b',
                'pritchard.Ic', 'pritchard.u', 'pritchard.v', 'pritchard.V',
                'pritchard.y']
    lc_files = [os.path.join('../phoebe-testlib/hv2241', lc_file) for lc_file in lc_files]
    
    for lc_file in lc_files:
        check_parse_with_numpy(lc_file, type='lc')
    
    
    
def test_parse_rv_hv2241():
    
    rv_files = ['hv2241.final.rv1', 'hv2241.final.rv2']
    rv_files = [os.path.join('../phoebe-testlib/hv2241', rv_file) for rv_file in rv_files]
    
    for rv_file in rv_files:
        check_parse_with_numpy(rv_file, type='rv')


def test_parse_lc_GKDra():
    
    lc_files = ['GKDra.B', 'GKDra.V']
    lc_files = [os.path.join('../phoebe-testlib/GKDra', lc_file) for lc_file in lc_files]
    
    for lc_file in lc_files:
        check_parse_with_numpy(lc_file, type='lc')

def test_parse_rv_GKDra():
    
    rv_files = ['GKDra.rv1', 'GKDra.rv1']
    rv_files = [os.path.join('../phoebe-testlib/GKDra', rv_file) for rv_file in rv_files]
    
    for rv_file in rv_files:
        check_parse_with_numpy(rv_file, type='rv')

def test_parse_lc_ABCas():
    
    lc_files = ['ABCas.y', 'ABCas.b', 'ABCas.u', 'ABCas.v']
    lc_files = [os.path.join('../phoebe-testlib/ABCas', lc_file) for lc_file in lc_files]
    
    for lc_file in lc_files:
        check_parse_with_numpy(lc_file, type='lc')


def test_parse_lc_hd174884():
    
    lc_files = ['hd174884.phased.data']
    lc_files = [os.path.join('../phoebe-testlib/HD174884', lc_file) for lc_file in lc_files]
    
    for lc_file in lc_files:
        check_parse_with_numpy(lc_file, type='lc')

def test_parse_rv_hd174884():
    
    rv_files = ['hd174884.rv1', 'hd174884.rv2']
    rv_files = [os.path.join('../phoebe-testlib/HD174884', rv_file) for rv_file in rv_files]
    
    for rv_file in rv_files:
        check_parse_with_numpy(rv_file, type='rv', columns = ['time', 'rv', 'sigma'])

def test_parse_phased_data():
    phasedfile = '../phoebe-testlib/HD174884/hd174884.phased.data'
    obs, pbdep = phoebe.parse_lc(phasedfile, columns=['phase', 'flux'])
    assert(len(obs[0]['time']) == 0)
    assert(obs[0]['columns'] == ['phase', 'flux', 'sigma'])

def test_parse_phased_data_mag():
    phasedfile = '../phoebe-testlib/HD174884/hd174884.phased.data'
    obs, pbdep = phoebe.parse_lc(phasedfile, columns=['phase', 'mag'])
    assert(len(obs[0]['time']) == 0)
    assert(obs[0]['columns'] == ['phase', 'flux', 'sigma'])
    
    data = np.loadtxt(phasedfile).T
    
    flux1 = phoebe.convert('mag','erg/s/cm2/AA',data[1])
    flux2 = phoebe.convert('W/m3','erg/s/cm2/AA',10**(-data[1]/2.5))
    assert(np.all(flux1 == obs[0]['flux']))
    assert(np.all(flux2 == obs[0]['flux']))
    

def test_parse_header():
    
    rv_files = ['hv2241.final.rv1', 'hv2241.final.rv2']
    lc_files = ['asas.V', 'davidge.B', 'davidge.U', 'davidge.V', 'pritchard.b',
                'pritchard.Ic', 'pritchard.u', 'pritchard.v', 'pritchard.V',
                'pritchard.y']
    rv_files = [os.path.join('../phoebe-testlib/hv2241', ifile) for ifile in rv_files]
    lc_files = [os.path.join('../phoebe-testlib/hv2241', ifile) for ifile in lc_files]
    
    for ff in rv_files:
        info, sets = phoebe.parameters.datasets.parse_header(ff, ext='rv')
        assert(info[2] in [2,3])
        assert(info[:2] == (None, None))
        assert(sets[0].get_context() == 'rvdep')
        assert(sets[1].get_context() == 'rvobs')
              
    for ff in lc_files:
        info, sets = phoebe.parameters.datasets.parse_header(ff, ext='lc')
        assert(info[2] in [2,3])
        assert(info[:2] == (None, None))
        assert(sets[0].get_context() == 'lcdep')
        assert(sets[1].get_context() == 'lcobs')
    
    filename = '../phoebe-testlib/datafiles/example0.rv'
    info, sets = phoebe.parameters.datasets.parse_header(filename, ext='rv')
    assert(info == (None, 'componentA', 3))
    
    filename = '../phoebe-testlib/datafiles/example1.rv'
    info, sets = phoebe.parameters.datasets.parse_header(filename, ext='rv')
    assert(info == (None, None, 3))
    
    filename = '../phoebe-testlib/datafiles/example2.rv'
    info, sets = phoebe.parameters.datasets.parse_header(filename, ext='rv')
    assert(info == (['rv', 'time', 'sigma'], ['componentA', 'componentA', 'componentA'], 3))
    
    filename = '../phoebe-testlib/datafiles/example3.rv'
    info, sets = phoebe.parameters.datasets.parse_header(filename, ext='rv')
    assert(info[0] == ['rv', 'time', 'sigma', 'sigma', 'rv'])
    assert(info[1] == ['HV2241a', 'None', 'HV2241a', 'HV2241b', 'HV2241b'])
    assert(info[2] == 5)
    
    filename = '../phoebe-testlib/datafiles/example1.phot'
    info, sets = phoebe.parameters.datasets.parse_header(filename, ext='rv')
    
    filename = '../phoebe-testlib/datafiles/example2.phot'
    info, sets = phoebe.parameters.datasets.parse_header(filename, ext='rv')
    
    filename = '../phoebe-testlib/datafiles/example3.phot'
    info, sets = phoebe.parameters.datasets.parse_header(filename, ext='rv')
    
    filename = '../phoebe-testlib/datafiles/example4.phot'
    info, sets = phoebe.parameters.datasets.parse_header(filename, ext='rv')
    
def test_parse_lc_01():
    
    filename = '../phoebe-testlib/datafiles/example0.rv'
    
    obs = phoebe.parse_rv(filename)
    
    time = np.array([2455453.0, 2455453.1, 2455453.2, 2455453.3])
    rv = np.array([-10., -5., 2., 6.])
    sigma = np.array([0.1, 0.15, 0.11, 0.09])
    
    assert(np.all(obs['componentA'][0][0]['time'] == time))
    assert(np.all(obs['componentA'][0][0]['rv'] == rv))
    assert(np.all(obs['componentA'][0][0]['sigma'] == sigma))
    assert(obs['componentA'][0][0]['ref'] == 'SiII_rvs')
    assert(obs['componentA'][1][0]['ref'] == 'SiII_rvs')
    assert(obs['componentA'][1][0]['passband'] == 'JOHNSON.B')
    assert(obs['componentA'][1][0]['atm'] == 'kurucz')
    