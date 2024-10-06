#!/usr/bin/env python3

import numpy as np
from phoebe.dynamics import coord_b2j
from phoebe.dynamics import orbel_xv2el

def invgeometry(m, rb, vb, geometry='hierarchical'):
    """
    Convert barycentric coordinates to elements for various geometries.
   
    """

    if geometry == 'hierarchical':

        _invgeometry = hierarchical

    elif geometry == 'twopairs':

        _invgeometry = twopairs

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

def hierarchical(m, rb, vb):
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
    rj, vj = coord_b2j.coord_b2j(m, rb, vb)

    # compute osculating elements
    msum = m[0]
    for j in range(1, nbod):
        msum += m[j]
        ialpha = -1

        elmts[j-1,:] = orbel_xv2el.orbel_xv2el(msum, rj[j], vj[j])
        euler[j,:] = orbel_xv2el.get_euler(msum, elmts[j-1])
        roche[j,:] = orbel_xv2el.get_roche(msum, elmts[j-1])

        if j==1:
            euler[0,:] = euler[1,:]
            roche[0,:] = roche[1,:]
        if j>=1:
            euler[j,0] += np.pi

    return elmts, euler, roche

def twopairs(m, rb, vb):
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

    elmts[0,:] = orbel_xv2el.orbel_xv2el(msum, r2_1, v2_1)
    euler[0,:] = orbel_xv2el.get_euler(msum, elmts[0])
    roche[0,:] = orbel_xv2el.get_roche(msum, elmts[0])

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

    elmts[1,:] = orbel_xv2el.orbel_xv2el(msum, r4_3, v4_3)
    euler[2,:] = orbel_xv2el.get_euler(msum, elmts[1])
    roche[2,:] = orbel_xv2el.get_roche(msum, elmts[1])
    
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
    elmts[2,:] = orbel_xv2el.orbel_xv2el(msum, r34_12, v34_12)

    # everything to Jacobian
    rj, vj = coord_b2j.coord_b2j(m, rb, vb)
    
    # other bodies (also Jacobian)
    for j in range(4, nbod):
        msum += m[j]

        elmts[j-1,:] = orbel_xv2el.orbel_xv2el(msum, rj[j], vj[j])
        euler[j,:] = orbel_xv2el.get_euler(msum, elmts[j-1])
        roche[j,:] = orbel_xv2el.get_roche(msum, elmts[j-1])

        euler[j,0] += np.pi

    return elmts, euler, roche

if __name__ == "__main__":

    day = 86400.
    au = 1.496e11
    M_S = 2.e30
    G = 6.67e-11
    gms = G*M_S / (au**3 * day**-2)
    vk = np.sqrt(gms/1.0)

    m = gms*np.array([1.0, 0.0])
    rb = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vb = np.array([[0.0, 0.0, 0.0], [0.0, vk, 0.0]])

    elmts, euler, roche = hierarchical(m, rb, vb)

    print("m = ", m)
    print("elmts[0] = ", elmts[0])
    print("euler[0] = ", euler[0])
    print("euler[1] = ", euler[1])
    print("roche[0] = ", roche[0])
    print("roche[1] = ", roche[1])

    m = gms*np.array([1.0, 1.0, 0.5, 0.5])
    rb = np.array([ \
        [-0.38333333, 0.0, 0.0], \
        [-0.28333333, 0.0, 0.0], \
        [ 0.56666667, 0.0, 0.0], \
        [ 0.76666667, 0.0, 0.0], \
        ])
    vb = np.array([ \
        [0.0, -0.04852087, 0.0], \
        [0.0,  0.02860663, 0.0], \
        [0.0,  0.00063236, 0.0], \
        [0.0,  0.03919611, 0.0], \
        ])

    elmts, euler, roche = twopairs(m, rb, vb)

    print("")
    print("m = ", m)
    print("elmts[0] = ", elmts[0])
    print("elmts[1] = ", elmts[1])
    print("elmts[2] = ", elmts[2])
    print("euler[0] = ", euler[0])
    print("euler[1] = ", euler[1])
    print("euler[2] = ", euler[2])
    print("euler[3] = ", euler[3])
    print("roche[0] = ", roche[0])
    print("roche[1] = ", roche[1])
    print("roche[2] = ", roche[2])
    print("roche[3] = ", roche[3])


