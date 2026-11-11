"""
Third light with l3_mode='fraction' in every pblum_mode (#1187: under 'dataset-scaled' it was 4*pi too
large for detached systems and 8*pi for contact systems).

l3_frac is the fraction of the total flux (system + third light). For spherical stars outside eclipse the
system flux is unambiguous (the integrated mesh flux, L/(4 pi d^2)), so there the share must be l3_frac in
every pblum mode. The distance is 10 pc: at PHOEBE's default of 1 m every distance factor would be 1.
Contact systems cannot be spherical; for them every mode must give the same share as component-coupled
(an envelope's pblum must not be counted on top of its halves).
One dataset per pblum mode in a single bundle and two times keep this fast.
"""

import functools

import pytest
import phoebe
from phoebe import u

L3_FRAC = 0.05
TIMES = [0.1, 0.4]    # outside both eclipses (spheres: P = 1 d, eclipses at 0 and 0.5)
# dataset label: (pblum_mode, pblum_dataset for 'dataset-coupled')
MODES = {'lc_cc': ('component-coupled', None),
         'lc_abs': ('absolute', None),
         'lc_dec': ('decoupled', None),
         'lc_dss': ('dataset-scaled', None),
         'lc_dcc': ('dataset-coupled', 'lc_cc'),
         'lc_dcs': ('dataset-coupled', 'lc_dss')}
# For contacts only the dataset-scaled paths (reference and coupled dataset) handle pblums per mode; the
# other modes share compute_pblums' pbflux and are covered by the spheres above.
CONTACT_MODES = ['lc_cc', 'lc_dss', 'lc_dcs']


def _ids(datasets):
    return [f'{MODES[ds][0]}-to-{MODES[ds][1]}' if MODES[ds][1] else MODES[ds][0] for ds in datasets]


@functools.lru_cache(maxsize=None)
def _shares(contact_binary):
    """Third-light share of the total flux per dataset. With f the system flux and s the dataset-scaling
    (flux_scale, 1 otherwise), m0 = s0*f and m1 = s1*(f + l3), so l3/(f + l3) = 1 - (m0/s0)/(m1/s1)."""
    datasets = CONTACT_MODES if contact_binary else list(MODES)
    b = phoebe.default_binary(contact_binary=contact_binary)
    for dataset in datasets:
        pblum_mode, pblum_dataset = MODES[dataset]
        kwargs = {'pblum_dataset': pblum_dataset} if pblum_dataset else {}
        if pblum_mode == 'dataset-scaled':
            # observations to scale to; any value will do, the third-light share must not depend on it
            kwargs.update(fluxes=[1.0, 1.0], sigmas=[1e-3, 1e-3])
        b.add_dataset('lc', times=TIMES, dataset=dataset, pblum_mode=pblum_mode, l3_mode='fraction',
                      ld_mode='manual', ld_func='linear', ld_coeffs=[0.5], **kwargs)
    for qualifier, value in (('atm', 'blackbody'), ('irrad_method', 'none'), ('ld_mode_bol', 'manual'),
                             ('ld_func_bol', 'linear'), ('ld_coeffs_bol', [0.5])):
        b.set_value_all(qualifier, value)
    if not contact_binary:
        b.set_value_all('distortion_method', 'sphere')
        b.set_value('distance', context='system', value=10*u.pc)

    unscaled = []
    for l3_frac in (0.0, L3_FRAC):
        b.set_value_all('l3_frac', l3_frac)
        b.run_compute(model='l3test', overwrite=True)
        model = b.get_model('l3test')
        unscaled.append({ds: model.get_value('fluxes', dataset=ds)[0] /
                         (model.get_value('flux_scale', dataset=ds) if 'flux_scale' in model.filter(dataset=ds).qualifiers else 1.0)
                         for ds in datasets})
    return {ds: 1 - unscaled[0][ds] / unscaled[1][ds] for ds in datasets}


@pytest.mark.parametrize('dataset', list(MODES), ids=_ids(MODES))
def test_l3_frac_spheres(dataset):
    share = _shares(False)[dataset]
    assert abs(share / L3_FRAC - 1) < 5e-3, share


@pytest.mark.parametrize('dataset', CONTACT_MODES[1:], ids=_ids(CONTACT_MODES[1:]))
def test_l3_frac_contact_matches_component_coupled(dataset):
    shares = _shares(True)
    assert abs(shares[dataset] / shares['lc_cc'] - 1) < 1e-6, (shares[dataset], shares['lc_cc'])


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    print('spheres', _shares(False))
    print('contact', _shares(True))
