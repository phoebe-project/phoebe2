#!/usr/bin/env python3

import numpy as np
from phoebe.dynamics import coord_h2j
from phoebe.dynamics import coord_j2b
from phoebe.dynamics import orbel_el2xv

def geometry(m, elmts, geometry='hierarchical'):
    """
    Convert elements to barycentric coordinates for various geometries.
   
    """

    if geometry == 'hierarchical':

        _geometry = hierarchical

    elif geometry == 'twopairs':

        _geometry = twopairs

    else:
        raise NotImplementedError

    nbod = len(m)
    xs = np.zeros((nbod))
    ys = np.zeros((nbod))
    zs = np.zeros((nbod))
    vxs = np.zeros((nbod))
    vys = np.zeros((nbod))
    vzs = np.zeros((nbod))

    rb, vb = _geometry(m, elmts)

    # orientation
    rb[:,0] *= -1.0
    rb[:,1] *= -1.0
    vb[:,0] *= -1.0
    vb[:,1] *= -1.0

    xs[:] = rb[:,0]
    ys[:] = rb[:,1]
    zs[:] = rb[:,2]
    vxs[:] = vb[:,0]
    vys[:] = vb[:,1]
    vzs[:] = vb[:,2]

    return xs, ys, zs, vxs, vys, vzs

def hierarchical(m, elmts):
    """
     _          \                |
    / \          |               |
    1 2          3               4
    \_/          |               |
                /                |
    """

    nbod = len(m)
    rj = np.zeros((nbod, 3))
    vj = np.zeros((nbod, 3))

    # compute Jacobi coordinates
    msum = m[0]
    for j in range(1, nbod):
        msum += m[j]
        ialpha = -1

        rj[j], vj[j] = orbel_el2xv.orbel_el2xv(msum, ialpha, elmts[j-1])

    # convert to barycentric frame
    rb, vb = coord_j2b.coord_j2b(m, rj, vj)

    return rb, vb

def twopairs(m, elmts):
    """
     _          _               \ 
    / \        / \               |
    1 2        3 4               5
    \_/        \_/               |
                                / 
    """

    nbod = len(m)
    rh = np.zeros((nbod, 3))
    vh = np.zeros((nbod, 3))
    rj = np.zeros((nbod, 3))
    vj = np.zeros((nbod, 3))

    # (1+2) pair, 1-centric coordinates
    msum = m[0]+m[1]
    ialpha = -1
    r2_1, v2_1 = orbel_el2xv.orbel_el2xv(msum, ialpha, elmts[0])

    # barycenter
    r12_1 = m[1]/msum * r2_1
    v12_1 = m[1]/msum * v2_1

    # (3+4) pair, 3-centric
    msum = m[2]+m[3]
    r4_3, v4_3 = orbel_el2xv.orbel_el2xv(msum, ialpha, elmts[1])

    # barycenter
    r34_3 = m[3]/msum * r4_3
    v34_3 = m[3]/msum * v4_3

    # (1+2)+(3+4) mutual orbit, (1+2)-centric
    msum = m[0]+m[1]+m[2]+m[3]
    r34_12, v34_12 = orbel_el2xv.orbel_el2xv(msum, ialpha, elmts[2])

    # everything to 1-centric
    rh[0,:] = 0.0
    vh[0,:] = 0.0
    rh[1,:] = r2_1
    vh[1,:] = v2_1
    rh[2,:] = r12_1 + r34_12 - r34_3
    vh[2,:] = v12_1 + v34_12 - v34_3
    rh[3,:] = rh[2,:] + r4_3
    vh[3,:] = vh[2,:] + v4_3

    # everything to Jacobian
    rj[0:4], vj[0:4] = coord_h2j.coord_h2j(m[0:4], rh[0:4], vh[0:4])

    # other bodies (also Jacobian)
    for j in range(4, nbod):
        msum += m[j]

        rj[j], vj[j] = orbel_el2xv.orbel_el2xv(msum, ialpha, elmts[j-1])

    # convert to barycentric frame
    rb, vb = coord_j2b.coord_j2b(m, rj, vj)

    return rb, vb

if __name__ == "__main__":

    day = 86400.
    au = 1.496e11
    M_S = 2.e30
    G = 6.67e-11
    gms = G*M_S / (au**3 * day**-2)

    m = gms*np.array([1.0, 0.0])
    elmts = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    rb, vb = hierarchical(m, elmts)

    print("m = ", m)
    print("rb[0] = ", rb[0])
    print("rb[1] = ", rb[1])
    print("vb[0] = ", vb[0])
    print("vb[1] = ", vb[1])

    m = gms*np.array([1.0, 1.0, 0.5, 0.5])
    elmts = []
    elmts.append([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    elmts.append([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    elmts.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    rb, vb = twopairs(m, elmts)

    print("")
    print("m = ", m)
    print("rb[0] = ", rb[0])
    print("rb[1] = ", rb[1])
    print("rb[2] = ", rb[2])
    print("rb[3] = ", rb[3])
    print("vb[0] = ", vb[0])
    print("vb[1] = ", vb[1])
    print("vb[2] = ", vb[2])
    print("vb[3] = ", vb[3])


