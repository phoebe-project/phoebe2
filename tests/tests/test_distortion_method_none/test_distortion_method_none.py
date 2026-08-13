"""
"""

import numpy as np
import pytest

import phoebe


def test_binary(plot=False):
    b = phoebe.Bundle.default_binary()
    b.add_dataset('lc', times=[0])
    b.add_dataset('lp', times=[0])
    b.add_dataset('rv', times=[0])
    b.add_dataset('mesh', compute_times=[0])

    for comp in ['primary', 'secondary']:
        other = 'secondary' if comp == 'primary' else 'primary'

        b.set_value_all('distortion_method', value='roche')
        b.set_value('distortion_method', component=comp, value='none')

        # the luminosity reference must be a component that actually has a
        # mesh, otherwise every flux is scaled by pblum/0.0 -> 0.0
        b.set_value_all('pblum_component', value=other)

        b.compute_pblums(pblum_method='stefan-boltzmann')
        b.compute_pblums()

        b.run_compute()

        fluxes = b.get_value(qualifier='fluxes', context='model')
        assert np.all(np.isfinite(fluxes))
        assert np.all(fluxes > 0)


def test_all_none_raises():
    # no component has a mesh, so there is nothing to integrate over and the
    # fluxes would come back as nans
    b = phoebe.Bundle.default_binary()
    b.add_dataset('lc', times=[0])
    b.set_value_all('distortion_method', value='none')

    report = b.run_checks_compute()
    assert not report.passed

    with pytest.raises(ValueError):
        b.run_compute()


def test_all_none_skip_checks_does_not_segfault():
    # regression test for the reflection code passing all-None per-body arrays
    # on to libphoebe (which dereferenced them as arrays and killed the kernel).
    # run_checks catches this case, so bypass it to exercise the backend guard.
    b = phoebe.Bundle.default_binary()
    b.add_dataset('lc', times=[0])
    b.set_value_all('distortion_method', value='none')

    assert b.get_value(qualifier='irrad_method', context='compute') != 'none'

    b.run_compute(skip_checks=True)

    # nothing has a mesh, so there is no flux to be had -- the point of this
    # test is purely that we get here at all rather than dying in libphoebe
    assert len(b.get_value(qualifier='fluxes', context='model')) == 1


def test_pblum_component_none_raises():
    # pblum_component defaults to 'primary', which here has no mesh and so no
    # luminosity to couple to -- every flux would be silently scaled to zero
    b = phoebe.Bundle.default_binary()
    b.add_dataset('lc', times=[0])
    b.set_value('distortion_method', component='primary', value='none')

    assert b.get_value(qualifier='pblum_mode', context='dataset') == 'component-coupled'
    assert b.get_value(qualifier='pblum_component', context='dataset') == 'primary'

    with pytest.raises(ValueError):
        b.run_compute()

    # ...but pointing pblum at the component that does have a mesh is fine
    b.set_value_all('pblum_component', value='secondary')
    b.run_compute()

    fluxes = b.get_value(qualifier='fluxes', context='model')
    assert np.all(np.isfinite(fluxes))
    assert np.all(fluxes > 0)


def test_pblum_decoupled_none_ok():
    # decoupled scales each component independently, so a meshless component
    # does not drag the other one to zero
    b = phoebe.Bundle.default_binary()
    b.add_dataset('lc', times=[0])
    b.set_value('distortion_method', component='primary', value='none')
    b.set_value_all('pblum_mode', value='decoupled')

    b.run_compute()

    fluxes = b.get_value(qualifier='fluxes', context='model')
    assert np.all(np.isfinite(fluxes))
    assert np.all(fluxes > 0)


def test_stale_none_with_wd_meshing_ok():
    # mesh_method='wd' ignores distortion_method and always meshes as roche, so
    # a 'none' left behind from marching must not trip the checks above
    phoebe.devel_on()  # required for wd meshing
    try:
        b = phoebe.Bundle.default_binary()
        b.add_dataset('lc', times=[0])
        b.set_value_all('distortion_method', value='none')  # while still visible
        b.set_value_all('mesh_method', value='wd')

        assert b.run_checks_compute().passed
    finally:
        phoebe.devel_off()  # reset for future tests


def test_all_none_dynamical_rv_ok():
    # dynamical RVs need no mesh at all, so distortion_method='none' everywhere
    # must remain a valid (if unusual) configuration
    b = phoebe.Bundle.default_binary()
    b.add_dataset('rv', times=[0, 0.25])
    b.set_value_all('distortion_method', value='none')
    b.set_value_all('rv_method', value='dynamical')

    b.run_compute()

    for comp in ['primary', 'secondary']:
        rvs = b.get_value(qualifier='rvs', component=comp, context='model')
        assert np.all(np.isfinite(rvs))


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_binary(plot=True)
    test_all_none_raises()
    test_all_none_skip_checks_does_not_segfault()
    test_pblum_component_none_raises()
    test_pblum_decoupled_none_ok()
    test_stale_none_with_wd_meshing_ok()
    test_all_none_dynamical_rv_ok()
