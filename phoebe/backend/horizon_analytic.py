import numpy as np

import libphoebe
from phoebe.backend.mesh import euler_trans_matrix, transform_position_array

import logging

logger = logging.getLogger("HORIZON_ANALYTIC")


def cartesian_to_polar(x, y, x0=0.0, y0=0.0):

    rhos = np.sqrt((x-x0)**2, (y-y0)**2)
    thetas = np.arctan2(y-y0, x-x0)

    return rhos, thetas


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


def wd(xs, ys, zs):
    pass