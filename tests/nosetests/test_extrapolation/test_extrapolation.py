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
    extrapolation_method = 'none'

    raise_on_nans = True
    with pytest.raises(ValueError):
        inorm = ndp.interp(query_pts, extrapolation_method=extrapolation_method, raise_on_nans=raise_on_nans)

    raise_on_nans = False
    inorm = ndp.interp(query_pts, extrapolation_method=extrapolation_method, raise_on_nans=raise_on_nans).flatten()
    print(f'{extrapolation_method=}, {raise_on_nans=}, {inorm=}: ')
    assert np.isnan(inorm[0]) and np.isnan(inorm[2])

    extrapolation_method = 'nearest'

    inorm = ndp.interp(query_pts, extrapolation_method='nearest', raise_on_nans=False).flatten()
    print(f'{extrapolation_method=}, {raise_on_nans=}, {inorm=}: ')

if __name__ == '__main__':
    test_extrapolation()
