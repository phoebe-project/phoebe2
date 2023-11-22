"""
"""

import phoebe
from phoebe import u
import numpy as np

def test_sun(plot=False):
    b = phoebe.default_star(starA='sun')

    b.set_value('teff', 1.0*u.solTeff)
    b.set_value('requiv', 1.0*u.solRad)
    b.set_value('mass', 1.0*u.solMass)
    b.set_value('period', 24.47*u.d)

    b.set_value('incl', 23.5*u.deg)
    b.set_value('distance', 1.0*u.AU)

    assert(b.get_value('teff', u.K)==5772)
    assert(b.get_value('requiv', u.solRad)==1.0)
    assert(b.get_value('mass', u.solMass)==1.0)
    assert(b.get_value('period', u.d)==24.47)
    assert(b.get_value('incl', u.deg)==23.5)
    assert(b.get_value('distance', u.m)==1.0*u.AU.to(u.m))

    b.add_dataset('lc', pblum=1*u.solLum)
    b.add_dataset('mesh', times=[0], columns=['teffs', 'areas', 'volume'], dataset='mesh01')

    b.run_compute(irrad_method='none', distortion_method='rotstar')

    #print abs(b.get_value('teffs', dataset='pbmesh').mean()-b.get_value('teff', context='component'))

    assert(abs(np.average(b.get_value('teffs', dataset='mesh01'), weights=b.get_value('areas', dataset='mesh01')) - b.get_value('teff', context='component')) < 1e-6)
    assert(abs(b.get_value('volume', dataset='mesh01')-4./3*np.pi*b.get_value('requiv', context='component')**3) < 1e-6)

    #print abs(b.get_value('fluxes@model')[0]-1357.12228578)

    if plot:
        axs, artists = b['mesh01'].plot(facecolor='teffs', show=True)

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    b = test_sun(plot=True)
