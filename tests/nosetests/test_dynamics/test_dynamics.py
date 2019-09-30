"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt


def _keplerian_v_nbody(b, plot=False):
    """
    test a single bundle for the phoebe backend's kepler vs nbody dynamics methods
    """

    # TODO: loop over ltte=True,False (once keplerian dynamics supports the switch)

    # b.add_compute(dynamics_method='bs')
    b.set_value('dynamics_method', 'bs')

    times = np.linspace(0, 100, 10000)
    nb_ts, nb_us, nb_vs, nb_ws, nb_vus, nb_vvs, nb_vws = phoebe.dynamics.nbody.dynamics_from_bundle(b, times, ltte=False)
    k_ts, k_us, k_vs, k_ws, k_vus, k_vvs, k_vws = phoebe.dynamics.keplerian.dynamics_from_bundle(b, times)

    assert(np.allclose(nb_ts, k_ts, 1e-8))
    for ci in range(len(b.hierarchy.get_stars())):
        # TODO: make atol lower (currently 1e-5 solRad which is awfully big, but 1e-6 currently fails!)
        if plot:
            print("max atol xs:", nb_us[ci] - k_us[ci])
            print("max atol ys:", nb_vs[ci] - k_vs[ci])
            print("max atol zs:", nb_ws[ci] - k_ws[ci])
            print("max atol vxs:", nb_vus[ci] - k_vus[ci])
            print("max atol vys:", nb_vvs[ci] - k_vvs[ci])
            print("max atol vzs:", nb_vws[ci] - k_vws[ci])

        assert(np.allclose(nb_us[ci], k_us[ci], atol=1e-5))
        assert(np.allclose(nb_vs[ci], k_vs[ci], atol=1e-5))
        assert(np.allclose(nb_ws[ci], k_ws[ci], atol=1e-5))
        assert(np.allclose(nb_vus[ci], k_vus[ci], atol=1e-4))
        assert(np.allclose(nb_vvs[ci], k_vvs[ci], atol=1e-4))
        assert(np.allclose(nb_vws[ci], k_vws[ci], atol=1e-4))

def _phoebe_v_photodynam(b, plot=False):
    """
    test a single bundle for phoebe's nbody vs photodynam via the frontend
    """

    times = np.linspace(0, 100, 1000)

    b.add_dataset('orb', times=times, dataset='orb01', component=b.hierarchy.get_stars())
    # photodynam and phoebe should have the same nbody defaults... if for some reason that changes,
    # then this will probably fail
    b.add_compute('photodynam', compute='pdcompute')
    # photodynam backend ONLY works with ltte=True, so we will run the phoebe backend with that as well
    # TODO: remove distortion_method='nbody' once that is supported
    # NOTE: bs is the exact same as that used in photodynam.  Nbody and rebound are slightly different.
    b.set_value('dynamics_method', 'bs')
    b.set_value('ltte', True)

    b.run_compute('pdcompute', model='pdresults')
    b.run_compute('phoebe01', model='phoeberesults')

    for comp in b.hierarchy.get_stars():
        # TODO: check to see how low we can make atol (or change to rtol?)
        # TODO: look into justification of flipping x and y for both dynamics (photodynam & phoebe)
        # TODO: why the small discrepancy (visible especially in y, still <1e-11) - possibly a difference in time0 or just a precision limit in the photodynam backend since loading from a file??


        if plot:
            for k in ['us', 'vs', 'ws', 'vus', 'vvs', 'vws']:
                plt.cla()
                plt.plot(b.get_value('times', model='phoeberesults', component=comp, unit=u.d), b.get_value(k, model='phoeberesults', component=comp), 'r-')
                plt.plot(b.get_value('times', model='phoeberesults', component=comp, unit=u.d), b.get_value(k, model='pdresults', component=comp), 'b-')
                diff = abs(b.get_value(k, model='phoeberesults', component=comp) - b.get_value(k, model='pdresults', component=comp))
                print("*** max abs ({}): {}".format(k, max(diff)))
                plt.xlabel('t')
                plt.ylabel(k)
                plt.show()

        assert(np.allclose(b.get_value('times', dataset='orb01', model='phoeberesults', component=comp, unit=u.d), b.get_value('times', dataset='orb01', model='pdresults', component=comp, unit=u.d), atol=1e-6))
        assert(np.allclose(b.get_value('us', dataset='orb01', model='phoeberesults', component=comp, unit=u.AU), b.get_value('us', dataset='orb01', model='pdresults', component=comp, unit=u.AU), atol=1e-6))
        assert(np.allclose(b.get_value('vs', dataset='orb01', model='phoeberesults', component=comp, unit=u.AU), b.get_value('vs', dataset='orb01', model='pdresults', component=comp, unit=u.AU), atol=1e-6))
        assert(np.allclose(b.get_value('ws', dataset='orb01', model='phoeberesults', component=comp, unit=u.AU), b.get_value('ws', dataset='orb01', model='pdresults', component=comp, unit=u.AU), atol=1e-6))
        assert(np.allclose(b.get_value('vus', dataset='orb01', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vus', dataset='orb01', model='pdresults', component=comp, unit=u.solRad/u.d), atol=1e-6))
        assert(np.allclose(b.get_value('vvs', dataset='orb01', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vvs', dataset='orb01', model='pdresults', component=comp, unit=u.solRad/u.d), atol=1e-6))
        assert(np.allclose(b.get_value('vws', dataset='orb01', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vws', dataset='orb01', model='pdresults', component=comp, unit=u.solRad/u.d), atol=1e-6))

def _frontend_v_backend(b, plot=False):
    """
    test a single bundle for the frontend vs backend access to both kepler and nbody dynamics
    """

    # TODO: loop over ltte=True,False

    times = np.linspace(0, 100, 21)
    b.add_dataset('orb', times=times, dataset='orb01', component=b.hierarchy.get_stars())
    b.rename_compute('phoebe01', 'nbody')
    b.set_value('dynamics_method', 'bs')

    b.add_compute('phoebe', dynamics_method='keplerian', compute='keplerian')

    # NBODY
    # do backend Nbody
    b_ts, b_us, b_vs, b_ws, b_vus, b_vvs, b_vws = phoebe.dynamics.nbody.dynamics_from_bundle(b, times, compute='nbody')

    # do frontend Nbody
    b.run_compute('nbody', model='nbodyresults')


    for ci,comp in enumerate(b.hierarchy.get_stars()):
        # TODO: can we lower tolerance?
        assert(np.allclose(b.get_value('times', dataset='orb01', model='nbodyresults', component=comp, unit=u.d), b_ts, atol=1e-6))
        assert(np.allclose(b.get_value('us', dataset='orb01', model='nbodyresults', component=comp, unit=u.solRad), b_us[ci], atol=1e-5))
        assert(np.allclose(b.get_value('vs', dataset='orb01', model='nbodyresults', component=comp, unit=u.solRad), b_vs[ci], atol=1e-5))
        assert(np.allclose(b.get_value('ws', dataset='orb01', model='nbodyresults', component=comp, unit=u.solRad), b_ws[ci], atol=1e-5))
        assert(np.allclose(b.get_value('vus', dataset='orb01', model='nbodyresults', component=comp, unit=u.solRad/u.d), b_vus[ci], atol=1e-4))
        assert(np.allclose(b.get_value('vvs', dataset='orb01', model='nbodyresults', component=comp, unit=u.solRad/u.d), b_vvs[ci], atol=1e-4))
        assert(np.allclose(b.get_value('vws', dataset='orb01', model='nbodyresults', component=comp, unit=u.solRad/u.d), b_vws[ci], atol=1e-4))




    # KEPLERIAN
    # do backend keplerian
    b_ts, b_xs, b_ys, b_zs, b_vxs, b_vys, b_vzs = phoebe.dynamics.keplerian.dynamics_from_bundle(b, times, compute='keplerian')


    # do frontend keplerian
    b.run_compute('keplerian', model='keplerianresults')


    # TODO: loop over components and assert
    for ci,comp in enumerate(b.hierarchy.get_stars()):
        # TODO: can we lower tolerance?
        assert(np.allclose(b.get_value('times', dataset='orb01', model='keplerianresults', component=comp, unit=u.d), b_ts, atol=1e-6))
        assert(np.allclose(b.get_value('us', dataset='orb01', model='keplerianresults', component=comp, unit=u.solRad), b_us[ci], atol=1e-5))
        assert(np.allclose(b.get_value('vs', dataset='orb01', model='keplerianresults', component=comp, unit=u.solRad), b_vs[ci], atol=1e-5))
        assert(np.allclose(b.get_value('ws', dataset='orb01', model='keplerianresults', component=comp, unit=u.solRad), b_ws[ci], atol=1e-5))
        assert(np.allclose(b.get_value('vus', dataset='orb01', model='keplerianresults', component=comp, unit=u.solRad/u.d), b_vus[ci], atol=1e-4))
        assert(np.allclose(b.get_value('vvs', dataset='orb01', model='keplerianresults', component=comp, unit=u.solRad/u.d), b_vvs[ci], atol=1e-4))
        assert(np.allclose(b.get_value('vws', dataset='orb01', model='keplerianresults', component=comp, unit=u.solRad/u.d), b_vws[ci], atol=1e-4))



def test_binary(plot=False):
    """
    """
    phoebe.devel_on() # required for nbody dynamics

    # TODO: grid over orbital parameters
    # TODO: once ps.copy is implemented, just send b.copy() to each of these

    b = phoebe.default_binary()
    b.get_parameter('dynamics_method')._choices = ['keplerian', 'bs']
    _keplerian_v_nbody(b, plot=plot)

    b = phoebe.default_binary()
    b.get_parameter('dynamics_method')._choices = ['keplerian', 'bs']
    _phoebe_v_photodynam(b, plot=plot)

    b = phoebe.default_binary()
    b.get_parameter('dynamics_method')._choices = ['keplerian', 'bs']
    _frontend_v_backend(b, plot=plot)

    phoebe.devel_off() # reset for future tests


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    test_binary(plot=True)

    # TODO: create tests for both triple configurations (A--B-C, A-B--C) - these should first be default bundles
