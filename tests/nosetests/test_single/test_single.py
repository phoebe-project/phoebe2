"""
"""

import phoebe
from phoebe import u
import numpy as np

def test_sun(plot=False):
    b = phoebe.default_star(starA='sun')

    b.set_value('teff', 1.0*u.solTeff)
    b.set_value('rpole', 1.0*u.solRad)
    b.set_value('mass', 1.0*u.solMass)
    b.set_value('period', 24.47*u.d)

    b.set_value('incl', 23.5*u.deg)
    b.set_value('distance', 1.0*u.AU)

    assert(b.get_value('teff', u.K)==5772)
    assert(b.get_value('rpole', u.solRad)==1.0)
    assert(b.get_value('mass', u.solMass)==1.0)
    assert(b.get_value('period', u.d)==24.47)
    assert(b.get_value('incl', u.deg)==23.5)
    assert(b.get_value('distance', u.m)==1.0*u.AU.to(u.m))

    b.add_dataset('lc', pblum=1*u.solLum)
    
    b.run_compute(protomesh=True, pbmesh=True, irrad_method='none', distortion_method='rotstar')
       
    #print abs(b.get_value('teffs', dataset='pbmesh').mean()-b.get_value('teff', context='component'))
    
    assert(abs(np.average(b.get_value('teffs', dataset='pbmesh'), weights=b.get_value('areas', dataset='pbmesh')) - b.get_value('teff', context='component')) < 1e-6)
    assert(abs(b.get_value('rpole', dataset='pbmesh')-b.get_value('rpole', context='component')) < 1e-6)
    
    #print abs(b.get_value('fluxes@model')[0]-1357.12228578)
    
    if plot:
        axs, artists = b['protomesh'].plot(facecolor='teffs', show=True)
        axs, artists = b['pbmesh'].plot(facecolor='teffs', show=True)

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_sun(plot=True)
