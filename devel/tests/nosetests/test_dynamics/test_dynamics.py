"""
"""

import phoebe2
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt


def _keplerian_v_nbody(b, plot=False):
    """
    test a single bundle for the phoebe backend's kepler vs nbody dynamics methods
    """

    # TODO: loop over ltte=True,False (once keplerian dynamics supports the switch)

    times = np.linspace(0, 100, 10000)
    nb_ts, nb_xs, nb_ys, nb_zs, nb_vxs, nb_vys, nb_vzs = phoebe2.dynamics.nbody.dynamics_from_bundle(b, times, ltte=False)
    k_ts, k_xs, k_ys, k_zs, k_vxs, k_vys, k_vzs = phoebe2.dynamics.keplerian.dynamics_from_bundle(b, times)

    assert(np.allclose(nb_ts, k_ts, 1e-8))
    for ci in range(len(b.hierarchy.get_stars())):
        # TODO: make atol lower (currently 1e-5 AU which is awfully big, but 1e-6 currently fails!)
        assert(np.allclose(nb_xs[ci].to(u.AU).value, k_xs[ci].to(u.AU).value, atol=1e-5))
        assert(np.allclose(nb_ys[ci].to(u.AU).value, k_ys[ci].to(u.AU).value, atol=1e-5))
        assert(np.allclose(nb_zs[ci].to(u.AU).value, k_zs[ci].to(u.AU).value, atol=1e-5))
        assert(np.allclose(nb_vxs[ci].to(u.solRad/u.d).value, k_vxs[ci].to(u.solRad/u.d).value, atol=1e-5))
        assert(np.allclose(nb_vys[ci].to(u.solRad/u.d).value, k_vys[ci].to(u.solRad/u.d).value, atol=1e-5))
        assert(np.allclose(nb_vzs[ci].to(u.solRad/u.d).value, k_vzs[ci].to(u.solRad/u.d).value, atol=1e-5))

def _phoebe_v_photodynam(b, plot=False):
    """
    test a single bundle for phoebe's nbody vs photodynam via the frontend
    """

    times = np.linspace(0, 100, 1000)

    b.add_dataset('orb', time=times, dataset='orb01', components=b.hierarchy.get_stars())
    # photodynam and phoebe should have the same nbody defaults... if for some reason that changes,
    # then this will probably fail
    b.add_compute('photodynam', compute='pd')
    # photodynam backend ONLY works with ltte=True, so we will run the phoebe backend with that as well
    # TODO: remove distortion_method='nbody' once that is supported
    b.add_compute('phoebe', dynamics_method='nbody', ltte=True, compute='phoebe')

    b.run_compute('pd', model='pdresults')
    b.run_compute('phoebe', model='phoeberesults')

    for comp in b.hierarchy.get_stars():
        # TODO: check to see how low we can make atol (or change to rtol?)
        # TODO: look into justification of flipping x and y for both dynamics (photodynam & phoebe)
        # TODO: why the small discrepancy (visible especially in y, still <1e-11) - possibly a difference in time0 or just a precision limit in the photodynam backend since loading from a file??


        if plot:
            for k in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
                plt.cla()
                plt.plot(b.get_value('time', model='phoeberesults', component=comp, unit=u.d), b.get_value(k, model='phoeberesults', component=comp), 'r-')
                plt.plot(b.get_value('time', model='phoeberesults', component=comp, unit=u.d), b.get_value(k, model='pdresults', component=comp), 'b-')
                diff = abs(b.get_value(k, model='phoeberesults', component=comp) - b.get_value(k, model='pdresults', component=comp))
                print "*** max abs: {}".format(max(diff))
                plt.xlabel('t')
                plt.ylabel(k)
                plt.show()

        assert(np.allclose(b.get_value('time', dataset='orb01', model='phoeberesults', component=comp, unit=u.d), b.get_value('time', dataset='orb01', model='pdresults', component=comp, unit=u.d), atol=1e-6))
        assert(np.allclose(b.get_value('x', dataset='orb01', model='phoeberesults', component=comp, unit=u.AU), b.get_value('x', dataset='orb01', model='pdresults', component=comp, unit=u.AU), atol=1e-6))
        assert(np.allclose(b.get_value('y', dataset='orb01', model='phoeberesults', component=comp, unit=u.AU), b.get_value('y', dataset='orb01', model='pdresults', component=comp, unit=u.AU), atol=1e-6))
        assert(np.allclose(b.get_value('z', dataset='orb01', model='phoeberesults', component=comp, unit=u.AU), b.get_value('z', dataset='orb01', model='pdresults', component=comp, unit=u.AU), atol=1e-6))
        assert(np.allclose(b.get_value('vx', dataset='orb01', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vx', dataset='orb01', model='pdresults', component=comp, unit=u.solRad/u.d), atol=1e-6))
        assert(np.allclose(b.get_value('vy', dataset='orb01', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vy', dataset='orb01', model='pdresults', component=comp, unit=u.solRad/u.d), atol=1e-6))
        assert(np.allclose(b.get_value('vz', dataset='orb01', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vz', dataset='orb01', model='pdresults', component=comp, unit=u.solRad/u.d), atol=1e-6))

def _frontend_v_backend(b, plot=False):
    """
    test a single bundle for the frontend vs backend access to both kepler and nbody dynamics
    """

    # TODO: loop over ltte=True,False

    times = np.linspace(0, 100, 100)
    b.add_dataset('orb', time=times, dataset='orb01', components=b.hierarchy.get_stars())
    b.add_compute('phoebe', dynamics_method='keplerian', compute='keplerian')
    b.add_compute('phoebe', dynamics_method='nbody', compute='nbody')


    # NBODY
    # do backend Nbody
    b_ts, b_xs, b_ys, b_zs, b_vxs, b_vys, b_vzs = phoebe2.dynamics.nbody.dynamics_from_bundle(b, times)

    # do frontend Nbody
    b.run_compute('nbody', model='nbodyresults')


    for ci,comp in enumerate(b.hierarchy.get_stars()):
        # TODO: can we lower tolerance?
        assert(np.allclose(b.get_value('time', dataset='orb01', model='nbodyresults', component=comp, unit=u.d), b_ts, atol=1e-6))
        assert(np.allclose(b.get_value('x', dataset='orb01', model='nbodyresults', component=comp, unit=u.AU), b_xs[ci].to(u.AU).value, atol=1e-6))
        assert(np.allclose(b.get_value('y', dataset='orb01', model='nbodyresults', component=comp, unit=u.AU), b_ys[ci].to(u.AU).value, atol=1e-6))
        assert(np.allclose(b.get_value('z', dataset='orb01', model='nbodyresults', component=comp, unit=u.AU), b_zs[ci].to(u.AU).value, atol=1e-6))
        assert(np.allclose(b.get_value('vx', dataset='orb01', model='nbodyresults', component=comp, unit=u.solRad/u.d), b_vxs[ci].to(u.solRad/u.d).value, atol=1e-6))
        assert(np.allclose(b.get_value('vy', dataset='orb01', model='nbodyresults', component=comp, unit=u.solRad/u.d), b_vys[ci].to(u.solRad/u.d).value, atol=1e-6))
        assert(np.allclose(b.get_value('vz', dataset='orb01', model='nbodyresults', component=comp, unit=u.solRad/u.d), b_vzs[ci].to(u.solRad/u.d).value, atol=1e-6))




    # KEPLERIAN
    # do backend keplerian
    b_ts, b_xs, b_ys, b_zs, b_vxs, b_vys, b_vzs = phoebe2.dynamics.keplerian.dynamics_from_bundle(b, times)


    # do frontend keplerian
    b.run_compute('keplerian', model='keplerianresults')


    # TODO: loop over components and assert
    for ci,comp in enumerate(b.hierarchy.get_stars()):
        # TODO: can we lower tolerance?
        assert(np.allclose(b.get_value('time', dataset='orb01', model='keplerianresults', component=comp, unit=u.d), b_ts, atol=1e-6))
        assert(np.allclose(b.get_value('x', dataset='orb01', model='keplerianresults', component=comp, unit=u.AU), b_xs[ci].to(u.AU).value, atol=1e-6))
        assert(np.allclose(b.get_value('y', dataset='orb01', model='keplerianresults', component=comp, unit=u.AU), b_ys[ci].to(u.AU).value, atol=1e-6))
        assert(np.allclose(b.get_value('z', dataset='orb01', model='keplerianresults', component=comp, unit=u.AU), b_zs[ci].to(u.AU).value, atol=1e-6))
        assert(np.allclose(b.get_value('vx', dataset='orb01', model='keplerianresults', component=comp, unit=u.solRad/u.d), b_vxs[ci].to(u.solRad/u.d).value, atol=1e-6))
        assert(np.allclose(b.get_value('vy', dataset='orb01', model='keplerianresults', component=comp, unit=u.solRad/u.d), b_vys[ci].to(u.solRad/u.d).value, atol=1e-6))
        assert(np.allclose(b.get_value('vz', dataset='orb01', model='keplerianresults', component=comp, unit=u.solRad/u.d), b_vzs[ci].to(u.solRad/u.d).value, atol=1e-6))



def test_binary(plot=False):
    """
    """

    # TODO: grid over orbital parameters
    # TODO: once ps.copy is implemented, just send b.copy() to each of these

    b = phoebe2.Bundle.default_binary()
    _keplerian_v_nbody(b, plot=plot)

    b = phoebe2.Bundle.default_binary()
    _phoebe_v_photodynam(b, plot=plot)

    b = phoebe2.Bundle.default_binary()
    _frontend_v_backend(b, plot=plot)


if __name__ == '__main__':
    logger = phoebe2.utils.get_basic_logger()


    test_binary(plot=True)

    # TODO: create tests for both triple configurations (A--B-C, A-B--C) - these should first be default bundles