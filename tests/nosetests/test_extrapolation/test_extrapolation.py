import pytest

import phoebe
import numpy as np
from phoebe.dependencies.ndpolator import Cndpolator

pb = phoebe.get_passband('Johnson:V')

axes = pb.atm_axes['ck2004'][:-1]
grid = pb.atm_photon_grid['ck2004'][...,-1,:]

ndp = Cndpolator(axes, grid)

query_pts = np.array([
    [ 3400., 4.43, 0.0],  # ( 0, 9, -5)
    [ 3600., 4.43, 0.0],  # ( 1, 9, -5)
    [51000., 4.43, 0.0],  # (75, 9, -5)
])


def test_extrapolation():
    # test extrapolation_method='none':

    with pytest.raises(ValueError):
        inorm = ndp.interp(query_pts, extrapolation_method='none', raise_on_nans=True)

    inorm = ndp.interp(query_pts, extrapolation_method='none', raise_on_nans=False).flatten()
    assert np.isnan(inorm[0]) and np.isnan(inorm[2])

if __name__ == '__main__':
    test_extrapolation()
