"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt


def _keplerian_v_nbody(b, ltte, period, plot=False):
    """
    test a single bundle for the phoebe backend's kepler vs nbody dynamics methods
    """

    # TODO: loop over ltte=True,False (once keplerian dynamics supports the switch)

    b.set_value('dynamics_method', 'bs')

    times = np.linspace(0, 5*period, 101)
    nb_ts, nb_us, nb_vs, nb_ws, nb_vus, nb_vvs, nb_vws = phoebe.dynamics.nbody.dynamics_from_bundle(b, times, ltte=ltte)
    k_ts, k_us, k_vs, k_ws, k_vus, k_vvs, k_vws = phoebe.dynamics.keplerian.dynamics_from_bundle(b, times, ltte=ltte)

    assert(np.allclose(nb_ts, k_ts, 1e-8))
    for ci in range(len(b.hierarchy.get_stars())):
        # TODO: make rtol lower if possible
        assert(np.allclose(nb_us[ci], k_us[ci], rtol=1e-5, atol=1e-2))
        assert(np.allclose(nb_vs[ci], k_vs[ci], rtol=1e-5, atol=1e-2))
        assert(np.allclose(nb_ws[ci], k_ws[ci], rtol=1e-5, atol=1e-2))

        # nbody ltte velocities are wrong so only check velocities if ltte off
        if not ltte:
            assert(np.allclose(nb_vus[ci], k_vus[ci], rtol=1e-5, atol=1e-2))
            assert(np.allclose(nb_vvs[ci], k_vvs[ci], rtol=1e-5, atol=1e-2))
            assert(np.allclose(nb_vws[ci], k_vws[ci], rtol=1e-5, atol=1e-2))

def _phoebe_v_photodynam(b, period, plot=False):
    """
    test a single bundle for phoebe's nbody vs photodynam via the frontend
    """

    times = np.linspace(0, 5*period, 21)

    b.add_dataset('orb', times=times, dataset='orb01', component=b.hierarchy.get_stars())
    # photodynam and phoebe should have the same nbody defaults... if for some reason that changes,
    # then this will probably fail
    b.add_compute('photodynam', compute='pdcompute')
    # photodynam backend ONLY works with ltte=True, so we will run the phoebe backend with that as well
    # TODO: remove distortion_method='nbody' once that is supported
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
                print("*** max abs: {}".format(max(diff)))
                plt.xlabel('t')
                plt.ylabel(k)
                plt.show()

        assert(np.allclose(b.get_value('times', model='phoeberesults', component=comp, unit=u.d), b.get_value('times', model='pdresults', component=comp, unit=u.d), rtol=0, atol=1e-05))
        assert(np.allclose(b.get_value('us', model='phoeberesults', component=comp, unit=u.AU), b.get_value('us', model='pdresults', component=comp, unit=u.AU), rtol=0, atol=1e-05))
        assert(np.allclose(b.get_value('vs', model='phoeberesults', component=comp, unit=u.AU), b.get_value('vs', model='pdresults', component=comp, unit=u.AU), rtol=0, atol=1e-05))
        assert(np.allclose(b.get_value('ws', model='phoeberesults', component=comp, unit=u.AU), b.get_value('ws', model='pdresults', component=comp, unit=u.AU), rtol=0, atol=1e-05))
        #assert(np.allclose(b.get_value('vxs', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vxs', model='pdresults', component=comp, unit=u.solRad/u.d), rtol=0, atol=1e-05))
        #assert(np.allclose(b.get_value('vys', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vys', model='pdresults', component=comp, unit=u.solRad/u.d), rtol=0, atol=1e-05))
        #assert(np.allclose(b.get_value('vzs', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vzs', model='pdresults', component=comp, unit=u.solRad/u.d), rtol=0, atol=1e-05))

def _frontend_v_backend(b, ltte, period, plot=False):
    """
    test a single bundle for the frontend vs backend access to both kepler and nbody dynamics
    """

    # TODO: loop over ltte=True,False

    times = np.linspace(0, 5*period, 101)
    b.add_dataset('orb', times=times, dataset='orb01', component=b.hierarchy.get_stars())
    b.rename_compute('phoebe01', 'nbody')
    b.set_value('dynamics_method', 'bs')
    b.set_value('ltte', ltte)

    b.add_compute('phoebe', dynamics_method='keplerian', compute='keplerian', ltte=ltte)



    # NBODY
    # do backend Nbody
    b_ts, b_us, b_vs, b_ws, b_vus, b_vvs, b_vws = phoebe.dynamics.nbody.dynamics_from_bundle(b, times, compute='nbody', ltte=ltte)

    # do frontend Nbody
    b.run_compute('nbody', model='nbodyresults')


    for ci,comp in enumerate(b.hierarchy.get_stars()):
        # TODO: can we lower tolerance?
        assert(np.allclose(b.get_value('times', model='nbodyresults', component=comp, unit=u.d), b_ts, rtol=0, atol=1e-6))
        assert(np.allclose(b.get_value('us', model='nbodyresults', component=comp, unit=u.solRad), b_us[ci], rtol=1e-7, atol=1e-4))
        assert(np.allclose(b.get_value('vs', model='nbodyresults', component=comp, unit=u.solRad), b_vs[ci], rtol=1e-7, atol=1e-4))
        assert(np.allclose(b.get_value('ws', model='nbodyresults', component=comp, unit=u.solRad), b_ws[ci], rtol=1e-7, atol=1e-4))
        if not ltte:
            assert(np.allclose(b.get_value('vus', model='nbodyresults', component=comp, unit=u.solRad/u.d), b_vus[ci], rtol=1e-7, atol=1e-4))
            assert(np.allclose(b.get_value('vvs', model='nbodyresults', component=comp, unit=u.solRad/u.d), b_vvs[ci], rtol=1e-7, atol=1e-4))
            assert(np.allclose(b.get_value('vws', model='nbodyresults', component=comp, unit=u.solRad/u.d), b_vws[ci], rtol=1e-7, atol=1e-4))

    # KEPLERIAN
    # do backend keplerian
    b_ts, b_us, b_vs, b_ws, b_vus, b_vvs, b_vws = phoebe.dynamics.keplerian.dynamics_from_bundle(b, times, compute='keplerian', ltte=ltte)

    # do frontend keplerian
    b.run_compute('keplerian', model='keplerianresults')

    for ci,comp in enumerate(b.hierarchy.get_stars()):
        # TODO: can we lower tolerance?
        assert(np.allclose(b.get_value('times', model='keplerianresults', component=comp, unit=u.d), b_ts, rtol=0, atol=1e-08))
        assert(np.allclose(b.get_value('us', model='keplerianresults', component=comp, unit=u.solRad), b_us[ci], rtol=0, atol=1e-08))
        assert(np.allclose(b.get_value('vs', model='keplerianresults', component=comp, unit=u.solRad), b_vs[ci], rtol=0, atol=1e-08))
        assert(np.allclose(b.get_value('ws', model='keplerianresults', component=comp, unit=u.solRad), b_ws[ci], rtol=0, atol=1e-08))
        assert(np.allclose(b.get_value('vus', model='keplerianresults', component=comp, unit=u.solRad/u.d), b_vus[ci], rtol=0, atol=1e-08))
        assert(np.allclose(b.get_value('vvs', model='keplerianresults', component=comp, unit=u.solRad/u.d), b_vvs[ci], rtol=0, atol=1e-08))
        assert(np.allclose(b.get_value('vws', model='keplerianresults', component=comp, unit=u.solRad/u.d), b_vws[ci], rtol=0, atol=1e-08))



def test_binary(plot=False):
    """
    """
    phoebe.devel_on() # required for nbody dynamics

    # TODO: once ps.copy is implemented, just send b.copy() to each of these

    # system = [sma (AU), period (d)]
    system1 = [0.05, 2.575]
    system2 = [1., 257.5]
    system3 = [40., 65000.]

    for system in [system1,system2,system3]:
        for q in [0.5,1.]:
            for ltte in [True, False]:
                print("test_dynamics_grid: sma={}, period={}, q={}, ltte={}".format(system[0], system[1], q, ltte))
                b = phoebe.default_binary()
                b.get_parameter('dynamics_method')._choices = ['keplerian', 'bs']

                b.set_default_unit_all('sma', u.AU)
                b.set_default_unit_all('period', u.d)

                b.set_value('sma@binary',system[0])
                b.set_value('period@binary', system[1])
                b.set_value('q', q)

                _keplerian_v_nbody(b, ltte, system[1], plot=plot)
                _frontend_v_backend(b, ltte, system[1], plot=plot)

    phoebe.devel_off()  # reset for future tests

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')


    test_binary(plot=True)

    # TODO: create tests for both triple configurations (A--B-C, A-B--C) - these should first be default bundles
