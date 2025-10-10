import numpy as np
from phoebe import u
from phoebe.atmospheres.passbands import Passband
from phoebe.atmospheres.models import BlackbodyModelAtmosphere


def test_bb_interpolation_accuracy():
    # Create a BlackbodyModelAtmosphere instance
    bbatm = BlackbodyModelAtmosphere()

    # Generate a mock passband transmission with a quadratic profile:
    wls = np.linspace(500., 700., 100)
    trs = -(np.linspace(0., 1., 100) - 0.5) ** 2 + 1
    ptf = np.column_stack((wls, trs))

    # Create a Passband instance
    pb = Passband(
        ptf=ptf,
        pbset='test',
        pbname='quad',
        wlunits=u.nm,
        calibrated=True
    )

    # Compute blackbody intensities:
    pb.compute_intensities(
        atm=bbatm,
        include_mus=False,
        include_extinction=False
    )

    # Check if the tables are correctly stored in the passband:
    assert 'blackbody' in pb.ndp, "Blackbody intensities not computed or stored in the passband."
    assert 'inorm@energy' in pb.ndp['blackbody'].tables, "Energy-weighted blackbody intensities table not found in the passband."
    assert 'inorm@photon' in pb.ndp['blackbody'].tables, "Photon-weighted blackbody intensities table not found in the passband."

    # Check if the interpolated intensities are properly gridded:
    assert 'grid' in pb.ndp['blackbody'].table['inorm@energy'], "Energy-weighted blackbody intensities not computed in the passband."
    grid = pb.ndp['blackbody'].table['inorm@energy']['grid']
    assert np.isfinite(grid).all(), "Energy-weighted blackbody intensities grid contains non-finite values."

    assert 'grid' in pb.ndp['blackbody'].table['inorm@photon'], "Photon-weighted blackbody intensities not computed in the passband."
    grid = pb.ndp['blackbody'].table['inorm@photon']['grid']
    assert np.isfinite(grid).all(), "Photon-weighted blackbody intensities grid contains non-finite values."

    assert len(pb.ndp['blackbody'].axes) == 1, "Blackbody ndpolator instance should have a single axis (teffs)."

    # Define test temperatures for interpolation:
    teffs = np.arange(3500., 499901., 100)

    # Compute theoretical blackbody intensities:
    bolseds = 2 * 6.62607015e-34 * 2.99792458e8**2 / pb.wl**5 / (np.exp(6.62607015e-34 * 2.99792458e8 / (pb.wl * 1.380649e-23 * teffs[:, None])) - 1)
    fltseds = bolseds * pb.ptf(pb.wl)  # Apply the passband transmission
    bbints = np.trapz(fltseds, pb.wl, axis=1) / pb.ptf_area
    bbints = np.log10(bbints).reshape(-1, 1)

    # Interpolate blackbody intensities from the table:
    query_pts = np.ascontiguousarray(teffs.reshape(-1, 1))
    interp_ints = pb.ndp['blackbody'].ndpolate('inorm@energy', query_pts=query_pts)['interps']

    assert interp_ints.shape == bbints.shape, "Interpolated intensities shape does not match theoretical blackbody intensities shape."
    np.testing.assert_allclose(
        interp_ints,
        bbints,
        rtol=1.2e-5,  # to be adjusted once we decide on the precision tolerance
        err_msg="Interpolated blackbody intensities do not match theoretical values."
    )


if __name__ == "__main__":
    test_bb_interpolation_accuracy()
