import numpy as np
import phoebe
import pytest


def _build_dc_bundle(times_primary, rvs_primary, sigmas_primary,
                     times_secondary, rvs_secondary, sigmas_secondary):
    b = phoebe.default_binary()
    b.add_solver('differential_corrections', solver='dc')

    b.add_dataset(
        kind='rv',
        dataset='rv01',
        compute_phases=np.linspace(0.0, 1.0, 31),
        times={'primary': times_primary, 'secondary': times_secondary},
        rvs={'primary': rvs_primary, 'secondary': rvs_secondary},
        sigmas={'primary': sigmas_primary, 'secondary': sigmas_secondary},
    )

    b.set_value(
        twig='fit_parameters@dc@differential_corrections@solver',
        value=['incl@binary'],
    )
    b.set_value(
        twig='steps@dc@differential_corrections@solver',
        value={'incl@binary': 0.01},
    )

    return b


@pytest.mark.parametrize(
    'case, times_primary, rvs_primary, sigmas_primary, times_secondary, rvs_secondary, sigmas_secondary, should_raise',
    [
        ('sb0', [], [], [], [], [], [], True),
        (
            'sb1_primary',
            np.linspace(0.0, 1.0, 20),
            20.0 * np.sin(2.0 * np.pi * np.linspace(0.0, 1.0, 20)),
            np.ones(20),
            [],
            [],
            [],
            False,
        ),
        (
            'sb1_secondary',
            [],
            [],
            [],
            np.linspace(0.0, 1.0, 20),
            -20.0 * np.sin(2.0 * np.pi * np.linspace(0.0, 1.0, 20)),
            np.ones(20),
            False,
        ),
        (
            'sb2',
            np.linspace(0.0, 1.0, 20),
            20.0 * np.sin(2.0 * np.pi * np.linspace(0.0, 1.0, 20)),
            np.ones(20),
            np.linspace(0.0, 1.0, 20),
            -20.0 * np.sin(2.0 * np.pi * np.linspace(0.0, 1.0, 20)),
            np.ones(20),
            False,
        ),
    ],
)
def test_dc_rv_sb0_sb1_sb2(case, times_primary, rvs_primary, sigmas_primary,
                           times_secondary, rvs_secondary, sigmas_secondary, should_raise):
    b = _build_dc_bundle(
        times_primary=times_primary,
        rvs_primary=rvs_primary,
        sigmas_primary=sigmas_primary,
        times_secondary=times_secondary,
        rvs_secondary=rvs_secondary,
        sigmas_secondary=sigmas_secondary,
    )

    if should_raise:
        with pytest.raises(
            ValueError,
            match='no observations found in enabled lc/rv datasets for differential corrections',
        ):
            b.run_solver(solver='dc', progressbar=False)
    else:
        b.run_solver(solver='dc', progressbar=False)
