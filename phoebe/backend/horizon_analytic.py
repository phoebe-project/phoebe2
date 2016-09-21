import numpy as np

import libphoebe
from phoebe.backend.mesh import euler_trans_matrix, transform_position_array
from phoebe.frontend import io


try:
    import phoebeBackend as phb
except ImportError:
    _can_phb = False
else:
    _can_phb = True
    _phb_init = False

import logging

logger = logging.getLogger("HORIZON_ANALYTIC")


def cartesian_to_polar(x, y, x0=0.0, y0=0.0):

    rhos = np.sqrt((x-x0)**2 + (y-y0)**2)
    thetas = np.arctan2(y-y0, x-x0)

    return rhos, thetas

def polar_to_cartesian(rho, theta, x0=0.0, y0=0.0):
    x = rho * np.cos(theta) + x0
    y = rho * np.sin(theta) + y0

    return x, y

def marching(q, F, d, Phi, scale, euler, pos):
    """

    viewing_vector is in ROCHE coordinate system so not necessarily along z
    q, F, d, Phi should be body._mesh_args
    scale should be body._scale
    """

    euler = np.asarray(euler)
    pos = np.asarray(pos)

    # print "*** horizon_analytic.marching pos", pos
    # print "*** horizon_analytic.marching euler", euler

    viewing_vector = transform_position_array(array=np.array([0, 0, -1]),
                                              pos=np.array([0., 0., 0.]),
                                              euler=euler,
                                              is_normal=True,
                                              reverse=True)

    # print "*** horizon_analytic.marching viewing vector", viewing_vector

    # note: all of this is done in the ROCHE coordinate system.
    horizon_roche = libphoebe.roche_horizon(viewing_vector, q, F, d, Phi, choice=0)

    horizon_roche *= scale

    horizon_orbit = transform_position_array(array=horizon_roche,
                                             pos=pos,
                                             euler=euler,
                                             is_normal=False,
                                             reverse=False)

    horizon_rhos, horizon_thetas = cartesian_to_polar(horizon_orbit[:,0], horizon_orbit[:,1], pos[0], pos[1])

    return {'xs': horizon_orbit[:,0], 'ys': horizon_orbit[:,1], 'zs': horizon_orbit[:,2],
            'rhos': horizon_rhos, 'thetas': horizon_thetas}


def wd(b, time, scale, pos):



    def rho(theta, c, s):
        sum = 0.0
        for i in range(len(c)):
            sum += c[i]*np.cos(i*theta)+s[i]*np.sin(i*theta)
        return sum

    if not _can_phb:
        return {'xs': [], 'ys': [], 'zs': [],
                'rhos': [], 'zs': []}

    if not _phb_init:
        phb.init()
        phb.configure()


    # TODO: move this outside the loop into backends.py?
    # TODO: make this work without the need to add an LC dataset
    io.pass_to_legacy(b, 'change_to_tmpfile')

    phb.open('change_to_tmpfile')
    phb.setpar('phoebe_indep', 'Time (HJD)')
    combo = phb.lc((time,), 0, 0, 1)
    # phb.quit()

    if not len(combo[1]['theta']):
        return {'xs': [], 'ys': [], 'zs': [],
                'rhos': [], 'thetas': []}

    thetas = np.linspace(-np.pi, np.pi, 1001)
    rhos = rho(thetas, combo[1]['hAc'], combo[1]['hAs'])

    rhos *= scale

    xs, ys = polar_to_cartesian(rhos, thetas, pos[0], pos[1])

    return {'xs': xs, 'ys': ys, 'zs': np.ones(xs.shape)*pos[2],
            'rhos': rhos, 'thetas': thetas}

