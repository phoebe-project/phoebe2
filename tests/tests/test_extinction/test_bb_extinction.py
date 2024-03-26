import phoebe
import numpy as np
import matplotlib.pyplot as plt


def test_low_level_function():
    pb = phoebe.get_passband('Johnson:V')
    if 'blackbody:ext' not in pb.content:
        raise ValueError(f'extinction tables not found in {pb.pbset}:{pb.pbname}')

    ebvs = np.linspace(0, 1, 101)
    rvs = 3.1 * np.ones_like(ebvs)
    teffs = 5772 * np.ones_like(ebvs)

    query_pts = np.vstack((teffs, rvs, ebvs)).T

    iext = pb.interpolate_extinct(query_pts=query_pts, atm='blackbody', intens_weighting='photon', extrapolation_method='none')
    plt.xlabel('E(B-V)')
    plt.ylabel('$I_\mathrm{norm}$ (blackbody)')
    plt.plot(ebvs, iext, 'b-')
    plt.show()


test_low_level_function()
