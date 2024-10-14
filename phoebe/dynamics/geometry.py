#!/usr/bin/env python3

import numpy as np

from phoebe.dynamics import coord
from phoebe.dynamics import orbel

def geometry(m, elmts, geometry='hierarchical'):
    """
    Convert elements to barycentric coordinates for various geometries.
   
    """

    if geometry == 'hierarchical':

        _geometry = geometry_hierarchical

    elif geometry == 'twopairs':

        _geometry = geometry_twopairs

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

def geometry_hierarchical(m, elmts):
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

        rj[j], vj[j] = orbel.orbel_el2xv(msum, ialpha, elmts[j-1])

    # convert to barycentric frame
    rb, vb = coord.coord_j2b(m, rj, vj)

    return rb, vb

def geometry_twopairs(m, elmts):
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
    r2_1, v2_1 = orbel.orbel_el2xv(msum, ialpha, elmts[0])

    # barycenter
    r12_1 = m[1]/msum * r2_1
    v12_1 = m[1]/msum * v2_1

    # (3+4) pair, 3-centric
    msum = m[2]+m[3]
    r4_3, v4_3 = orbel.orbel_el2xv(msum, ialpha, elmts[1])

    # barycenter
    r34_3 = m[3]/msum * r4_3
    v34_3 = m[3]/msum * v4_3

    # (1+2)+(3+4) mutual orbit, (1+2)-centric
    msum = m[0]+m[1]+m[2]+m[3]
    r34_12, v34_12 = orbel.orbel_el2xv(msum, ialpha, elmts[2])

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
    rj[0:4], vj[0:4] = coord.coord_h2j(m[0:4], rh[0:4], vh[0:4])

    # other bodies (also Jacobian)
    for j in range(4, nbod):
        msum += m[j]

        rj[j], vj[j] = orbel.orbel_el2xv(msum, ialpha, elmts[j-1])

    # convert to barycentric frame
    rb, vb = coord.coord_j2b(m, rj, vj)

    return rb, vb

########################################################################

def invgeometry(m, rb, vb, geometry='hierarchical'):
    """
    Convert barycentric coordinates to elements for various geometries.
   
    """

    if geometry == 'hierarchical':

        _invgeometry = invgeometry_hierarchical

    elif geometry == 'twopairs':

        _invgeometry = invgeometry_twopairs

    else:
        raise NotImplementedError

    # cf. mutable numpy arrays
    rb = np.copy(rb)
    vb = np.copy(vb)

    # orientation
    rb[:,0] *= -1.0
    rb[:,1] *= -1.0
    vb[:,0] *= -1.0
    vb[:,1] *= -1.0

    return _invgeometry(m, rb, vb)

def invgeometry_hierarchical(m, rb, vb):
    """
     _          \                |
    / \          |               |
    1 2          3               4
    \_/          |               |
                /                |
    """

    nbod = len(m)
    elmts = np.zeros((nbod-1, 6))
    euler = np.zeros((nbod, 3))
    roche = np.zeros((nbod, 2))

    # convert to Jacobian coordinates
    rj, vj = coord.coord_b2j(m, rb, vb)

    # compute osculating elements
    msum = m[0]
    for j in range(1, nbod):
        msum += m[j]
        ialpha = -1

        elmts[j-1,:] = orbel.orbel_xv2el(msum, rj[j], vj[j])
        euler[j,:] = orbel.get_euler(elmts[j-1])
        roche[j,:] = orbel.get_roche(msum, elmts[j-1])

        if j==1:
            euler[0,:] = euler[1,:]
            roche[0,:] = roche[1,:]
        if j>=1:
            euler[j,0] += np.pi

    return elmts, euler, roche

def invgeometry_twopairs(m, rb, vb):
    """
     _          _               \ 
    / \        / \               |
    1 2        3 4               5
    \_/        \_/               |
                                / 
    """

    nbod = len(m)
    elmts = np.zeros((nbod-1, 6))
    euler = np.zeros((nbod, 3))
    roche = np.zeros((nbod, 2))

    # (1+2) pair, 1-centric coordinates
    msum = m[0]+m[1]
    r2_1 = rb[1] - rb[0]
    v2_1 = vb[1] - vb[0]

    elmts[0,:] = orbel.orbel_xv2el(msum, r2_1, v2_1)
    euler[0,:] = orbel.get_euler(elmts[0])
    roche[0,:] = orbel.get_roche(msum, elmts[0])

    euler[1,:] = euler[0,:]
    roche[1,:] = roche[0,:]
    euler[1,0] += np.pi

    # barycenter
    r12 = (m[0]*rb[0] + m[1]*rb[1])/msum
    v12 = (m[0]*vb[0] + m[1]*vb[1])/msum

    # (3+4) pair, 3-centric
    msum = m[2]+m[3]
    r4_3 = rb[3] - rb[2]
    v4_3 = vb[3] - vb[2]

    elmts[1,:] = orbel.orbel_xv2el(msum, r4_3, v4_3)
    euler[2,:] = orbel.get_euler(elmts[1])
    roche[2,:] = orbel.get_roche(msum, elmts[1])
    
    euler[3,:] = euler[2,:]
    roche[3,:] = roche[2,:]
    euler[3,0] += np.pi

    # barycenter
    r34 = (m[2]*rb[2] + m[3]*rb[3])/msum
    v34 = (m[2]*vb[2] + m[3]*vb[3])/msum

    # (1+2)+(3+4) mutual orbit, (1+2)-centric
    msum = m[0]+m[1]+m[2]+m[3]
    r34_12 = r34 - r12
    v34_12 = v34 - v12
    elmts[2,:] = orbel.orbel_xv2el(msum, r34_12, v34_12)

    # everything to Jacobian
    rj, vj = coord.coord_b2j(m, rb, vb)
    
    # other bodies (also Jacobian)
    for j in range(4, nbod):
        msum += m[j]

        elmts[j-1,:] = orbel.orbel_xv2el(msum, rj[j], vj[j])
        euler[j,:] = orbel.get_euler(elmts[j-1])
        roche[j,:] = orbel.get_roche(msum, elmts[j-1])

        euler[j,0] += np.pi

    return elmts, euler, roche


