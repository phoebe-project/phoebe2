"""
"""

import phoebe
import numpy as np


def test_aligned_static_mesh(verbose=False):
    # an aligned star in a circular orbit is meshed once; rounding in its spin axis
    # must not make it count as misaligned and be re-meshed at every time (#1178).
    # Which inclinations round that way depends on the machine, so each one is
    # tested together with the next float above it (physically the same system).
    incls = []
    for incl in [63., 70., 86.]:
        incls += [incl, float(np.nextafter(incl, 90.))]

    times = [0, 0.25, 0.5]
    b = phoebe.default_binary()
    b.add_dataset('mesh', compute_times=times, columns=['areas'], dataset='mesh01')

    for incl in incls:
        b.set_value('incl@binary', incl)
        b.run_compute(irrad_method='none', overwrite=True)
        for comp in ['primary', 'secondary']:
            areas0 = b.get_value(qualifier='areas', component=comp, time=times[0], context='model')
            for time in times[1:]:
                areas = b.get_value(qualifier='areas', component=comp, time=time, context='model')
                if verbose:
                    print(f"incl={incl!r} {comp} t={time}: same mesh: {np.array_equal(areas0, areas)}")
                assert np.array_equal(areas0, areas), f"{comp} re-meshed at incl={incl!r}"

    # a genuinely misaligned star is still re-meshed
    b.set_value('incl@binary', 86.)
    b.set_value('pitch@primary', 2)
    b.run_compute(irrad_method='none', overwrite=True)
    areas0 = b.get_value(qualifier='areas', component='primary', time=times[0], context='model')
    areas1 = b.get_value(qualifier='areas', component='primary', time=times[1], context='model')
    assert not np.array_equal(areas0, areas1)


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_aligned_static_mesh(verbose=True)
