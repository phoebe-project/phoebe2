"""
Unit tests for data parsers.
"""

import nose
import phoebe
import numpy as np
import os

basedir = os.path.dirname(os.path.abspath(__file__))
basedir = os.sep.join(basedir.split(os.sep)[:-1])



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
    assert(data.shape[0] == obs.shape[0])
    
    # We can't really test the number of columns, as the sigma column is
    # added if it's not there
    
    # Test if the columns contain the same data:
    columns = obs['columns']
    for col in range(min(len(columns),data.shape[1])):
        # Sigma column could have been added, so we don't test for it
        if columns[col] == 'sigma':
            continue
        assert(np.all(data[:,col] == obs[columns[col]]))
    
    
    
    
def test_parse_lc_hv2241():
    
    lc_files = ['asas.johnson_V.lc', 'davidge.johnson_B.lc',
                'davidge.johnson_U.lc', 'davidge.johnson_V.lc',
                'pritchard.stromgren_b.lc', 'pritchard.cousins_I.lc',
                'pritchard.stromgren_u.lc', 'pritchard.stromgren_v.lc',
                'pritchard.stromgren_v.lc', 'pritchard.stromgren_y.lc']
    lc_files = [os.path.join(basedir, 'hv2241', lc_file) for lc_file in lc_files]
    
    for lc_file in lc_files:
        check_parse_with_numpy(lc_file, type='lc')
    
    
    
def test_parse_rv_hv2241():
    
    rv_files = ['hv2241.final.rv1', 'hv2241.final.rv2']
    rv_files = [os.path.join(basedir, 'hv2241', rv_file) for rv_file in rv_files]
    
    for rv_file in rv_files:
        check_parse_with_numpy(rv_file, type='rv')


def test_parse_lc_GKDra():
    
    lc_files = ['GKDra.B', 'GKDra.V']
    lc_files = [os.path.join(basedir, 'GKDra', lc_file) for lc_file in lc_files]
    
    for lc_file in lc_files:
        check_parse_with_numpy(lc_file, type='lc')

def test_parse_rv_GKDra():
    
    rv_files = ['GKDra.rv1', 'GKDra.rv1']
    rv_files = [os.path.join(basedir, 'GKDra', rv_file) for rv_file in rv_files]
    
    for rv_file in rv_files:
        check_parse_with_numpy(rv_file, type='rv')

def test_parse_lc_ABCas():
    
    lc_files = ['ABCas.y', 'ABCas.b', 'ABCas.u', 'ABCas.v']
    lc_files = [os.path.join(basedir, 'ABCas', lc_file) for lc_file in lc_files]
    
    for lc_file in lc_files:
        check_parse_with_numpy(lc_file, type='lc')


def test_parse_lc_hd174884():
    
    lc_files = ['hd174884.phased.data']
    lc_files = [os.path.join(basedir, 'HD174884', lc_file) for lc_file in lc_files]
    
    for lc_file in lc_files:
        check_parse_with_numpy(lc_file, type='lc')

def test_parse_rv_hd174884():
    
    rv_files = ['hd174884.rv1', 'hd174884.rv2']
    rv_files = [os.path.join(basedir, 'HD174884', rv_file) for rv_file in rv_files]
    
    for rv_file in rv_files:
        check_parse_with_numpy(rv_file, type='rv', columns = ['time', 'rv', 'sigma'])

def test_parse_phased_data():
    phasedfile = os.path.join(basedir, 'HD174884/hd174884.phased.data')
    obs, pbdep = phoebe.parse_lc(phasedfile, columns=['phase', 'flux'])
    assert(len(obs['time']) == 0)
    assert(obs['columns'] == ['phase', 'flux', 'sigma'])

def test_parse_phased_data_mag():
    phasedfile = os.path.join(basedir, 'HD174884/hd174884.phased.data')
    obs, pbdep = phoebe.parse_lc(phasedfile, columns=['phase', 'mag'])
    assert(len(obs['time']) == 0)
    assert(obs['columns'] == ['phase', 'flux', 'sigma'])
    
    data = np.loadtxt(phasedfile).T
    
    #flux1 = phoebe.convert('mag','erg/s/cm2/AA',data[1], passband=pbdep['passband'])
    flux1 = phoebe.convert('mag','W/m3',data[1], passband=pbdep['passband'])
    #flux2 = phoebe.convert('W/m3','erg/s/cm2/AA',10**(-data[1]/2.5))
    assert(np.all(flux1 == obs['flux']))
    #assert(np.all(flux2 == obs['flux']))
    

def test_parse_header():
    
    info, sets = phoebe.parameters.datasets.parse_header(os.path.join(basedir, 'datafiles/example1.lc'))
    assert(info == (None, None, None, None, 2))
    
    info, sets = phoebe.parameters.datasets.parse_header(os.path.join(basedir, 'datafiles/example2.lc'))
    assert(info == (None, None, None, None, 4))
    
    info, sets = phoebe.parameters.datasets.parse_header(os.path.join(basedir, 'datafiles/example3.lc'))
    assert(info == (['time', 'flux', 'flag', 'sigma'], ['Vega', 'Vega', 'Vega', 'Vega'], None, None, 4))
    assert(sets[0]['passband'] == 'JOHNSON.B')
    assert(sets[0]['atm'] == 'kurucz')
    
    info, sets = phoebe.parameters.datasets.parse_header(os.path.join(basedir, 'datafiles/example4.lc'))
    assert(info == (['phase', 'mag', 'sigma', 'flag'], None, None, None, 4))
    assert(sets[0]['passband'] == 'JOHNSON.B')
    assert(sets[0]['atm'] == 'kurucz')
    
    info, sets = phoebe.parameters.datasets.parse_header(os.path.join(basedir, 'datafiles/example5.lc'))
    assert(info[0] == ['phase', 'flux', 'sigma', 'flag'])
    assert(info[1] == ['none', 'Vega', 'Vega', 'Vega'])
    assert(info[2]['phase'] == 'none')
    assert(info[2]['flux'] == 'mag')
    assert(info[2]['flag'] == 'none')
    assert(info[2]['sigma'] == 'mag')
    assert(info[3]['phase'] == 'f8')
    assert(info[3]['flux'] == 'f8')
    assert(info[3]['flag'] == 'int')
    assert(info[3]['sigma'] == 'f8')
    assert(info[4] == 4)
    assert(sets[0]['passband'] == 'JOHNSON.B')
    assert(sets[0]['atm'] == 'kurucz')
    
    
def test_parse_lc_01():
    
    obs, pbdep = phoebe.parse_lc(os.path.join(basedir, 'datafiles/example1.lc'))
    assert(obs['columns'] == ['time', 'flux', 'sigma'])
    assert(len(obs['time']) == 4)
    
    obs, pbdep = phoebe.parse_lc(os.path.join(basedir, 'datafiles/example2.lc'))
    assert(obs['columns'] == ['time', 'flux', 'sigma', 'flag'])
    assert(len(obs['time']) == 4)
    assert(pbdep['passband']=='JOHNSON.B')
    
    obs, pbdep = phoebe.parse_lc(os.path.join(basedir, 'datafiles/example3.lc'))
    assert(obs['columns'] == ['time', 'flux', 'flag', 'sigma'])
    assert(len(obs['time']) == 4)
    assert(np.all(obs['flag'] == 0))
    assert(pbdep['passband']=='JOHNSON.B')
    
    obs, pbdep = phoebe.parse_lc(os.path.join(basedir, 'datafiles/example4.lc'))
    assert(obs['columns'] == ['phase', 'flux', 'sigma', 'flag'])
    assert(len(obs['time']) == 0)
    assert(len(obs['phase']) == 4)
    assert(np.all(obs['flag'] == 0))
    assert(isinstance(obs['flag'][0], float))
    #bla = obs.get_value('flux','mag')
    #assert(np.all(bla-np.array([10., 11.1, 10.7, 10.8])<1e-14))
    assert(pbdep['passband']=='JOHNSON.B')
    
    obs, pbdep = phoebe.parse_lc(os.path.join(basedir, 'datafiles/example5.lc'))
    assert(obs['columns'] == ['phase', 'flux', 'sigma', 'flag'])
    assert(len(obs['time']) == 0)
    assert(len(obs['phase']) == 4)
    assert(np.all(obs['flag'] == 0))
    assert(isinstance(obs['flag'][0], int))
    #bla = obs.get_value('flux','mag')
    #assert(np.all(bla-np.array([10., 11.1, 10.7, 10.8])<1e-14))
    assert(pbdep['passband']=='JOHNSON.B')
    
    
if __name__=="__main__":
    
    test_parse_lc_01()
    test_parse_phased_data_mag()
    
    
    phasedfile = os.path.join(basedir, 'HD174884/hd174884.phased.data')
    obs, pbdep = phoebe.parse_lc(phasedfile, columns=['phase', 'flux'])
    
    assert(len(obs['time'])==0)
    assert(obs['columns']==['phase', 'flux', 'sigma'])
    
    
    phasedfile = os.path.join(basedir, 'HD174884/hd174884.phased.data')
    obs, pbdep = phoebe.parse_lc(phasedfile, columns=['phase', 'mag'])
    assert(len(obs['time']) == 0)
    assert(obs['columns'] == ['phase', 'flux', 'sigma'])
    
    data = np.loadtxt(phasedfile).T
    
    flux1 = phoebe.convert('mag','erg/s/cm2/AA',data[1], passband=pbdep['passband'])
    #flux2 = phoebe.convert('W/m3','erg/s/cm2/AA',10**(-data[1]/2.5))
    assert(np.all(flux1 == obs['flux']))