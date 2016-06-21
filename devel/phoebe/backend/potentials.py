import numpy as np

from math import sqrt, sin, cos, acos, atan2, trunc, pi
import os

import logging
logger = logging.getLogger("POTENTIALS")
logger.addHandler(logging.NullHandler())

def BinaryRoche (r, D, q, F, Omega=0.0):
    r"""
    Computes a value of the asynchronous, eccentric Roche potential.

    If :envvar:`Omega` is passed, it computes the difference.

    The asynchronous, eccentric Roche potential is given by [Wilson1979]_

    .. math::

        \Omega = \frac{1}{\sqrt{x^2 + y^2 + z^2}} + q\left(\frac{1}{\sqrt{(x-D)^2+y^2+z^2}} - \frac{x}{D^2}\right) + \frac{1}{2}F^2(1+q)(x^2+y^2)

    @param r:      relative radius vector (3 components)
    @type r: 3-tuple
    @param D:      instantaneous separation
    @type D: float
    @param q:      mass ratio
    @type q: float
    @param F:      synchronicity parameter
    @type F: float
    @param Omega:  value of the potential
    @type Omega: float
    """
    return 1.0/sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]) + q*(1.0/sqrt((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])-r[0]/D/D) + 0.5*F*F*(1+q)*(r[0]*r[0]+r[1]*r[1]) - Omega

def dBinaryRochedx (r, D, q, F):
    """
    Computes a derivative of the potential with respect to x.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return -r[0]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*(r[0]-D)*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5 -q/D/D + F*F*(1+q)*r[0]

def d2BinaryRochedx2(r, D, q, F):
    """
    Computes second derivative of the potential with respect to x.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return (2*r[0]*r[0]-r[1]*r[1]-r[2]*r[2])/(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**2.5 +\
          q*(2*(r[0]-D)*(r[0]-D)-r[1]*r[1]-r[2]*r[2])/((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**2.5 +\
          F*F*(1+q)

def dBinaryRochedy (r, D, q, F):
    """
    Computes a derivative of the potential with respect to y.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return -r[1]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*r[1]*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5 + F*F*(1+q)*r[1]

def  dBinaryRochedz(r, D, q, F):
    """
    Computes a derivative of the potential with respect to z.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """
    return -r[2]*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])**-1.5 -q*r[2]*((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**-1.5

def dBinaryRochedr (r, D, q, F):
    """
    Computes a derivative of the potential with respect to r.

    @param r:      relative radius vector (3 components)
    @param D:      instantaneous separation
    @param q:      mass ratio
    @param F:      synchronicity parameter
    """

    r2 = (r*r).sum()
    r1 = np.sqrt(r2)

    return -1./r2 - q*(r1-r[0]/r1*D)/((r[0]-D)*(r[0]-D)+r[1]*r[1]+r[2]*r[2])**1.5 - q*r[0]/r1/D/D + F*F*(1+q)*(1-r[2]*r[2]/r2)*r1

class MeshVertex:
    def __init__(self, r, dpdx, dpdy, dpdz, *args):
        """
        """
        # Normalized normal:
        #~ n = np.array([dpdx(r, *args), dpdy(r, *args), dpdz(r, *args)])
        #~ n /= np.sqrt(np.sum(n*n))
        nx = dpdx(r, *args)
        ny = dpdy(r, *args)
        nz = dpdz(r, *args)
        nn = sqrt(nx*nx+ny*ny+nz*nz)
        nx /= nn
        ny /= nn
        nz /= nn

        # Now we choose the first tangential direction. We have a whole
        # plane to choose from, so we'll choose something convenient.
        # The choice here is to set one tangential coordinate to 0. From
        # n \dot t=0: nx tx + ny ty = 0 => tx = -ny, ty = nx.
        # Since we are normalizing, we need to be careful that we don't
        # divide by a small number, hence the two prescriptions, either
        # for tz = 0 or for ty = 0.

        if nx > 0.5 or ny > 0.5:
            nn = sqrt(ny*ny+nx*nx)
            t1x = ny/nn
            t1y = -nx/nn
            t1z = 0.0
        else:
            nn = sqrt(nx*nx+nz*nz)
            t1x = -nz/nn
            t1y = 0.0
            t1z = nx/nn

        t2x = ny*t1z - nz*t1y
        t2y = nz*t1x - nx*t1z
        t2z = nx*t1y - ny*t1x

        self.r = r
        self.n = np.array((nx, ny, nz))
        self.t1 = np.array((t1x, t1y, t1z))
        self.t2 = np.array((t2x, t2y, t2z))

    def __repr__(self):
        repstr  = " r = (% 3.3f, % 3.3f, % 3.3f)\t" % (self.r[0],  self.r[1], self.r[2])
        repstr += " n = (% 3.3f, % 3.3f, % 3.3f)\t" % (self.n[0],  self.n[1], self.n[2])
        repstr += "t1 = (% 3.3f, % 3.3f, % 3.3f)\t" % (self.t1[0], self.t1[1], self.t1[2])
        repstr += "t2 = (% 3.3f, % 3.3f, % 3.3f)"   % (self.t2[0], self.t2[1], self.t2[2])
        return repstr

def project_onto_potential(r, pot_name, *args):
    """
    TODO: add documentation
    """

    pot = globals()[pot_name]
    dpdx = globals()['d%sdx'%(pot_name)]
    dpdy = globals()['d%sdy'%(pot_name)]
    dpdz = globals()['d%sdz'%(pot_name)]
    dpdr = globals()['d%sdr'%(pot_name)]

    n_iter = 0

    rmag, rmag0 = np.sqrt((r*r).sum()), 0
    lam, nu = r[0]/rmag, r[2]/rmag
    dc = np.array((lam, np.sqrt(1-lam*lam-nu*nu), nu)) # direction cosines -- must not change during reprojection
    D, q, F, p0 = args

    while np.abs(rmag-rmag0) > 1e-12 and n_iter < 100:
        rmag0 = rmag
        rmag = rmag0 - pot(rmag0*dc, *args)/dpdr(rmag0*dc, *args[:-1])
        n_iter += 1
    if n_iter == 100:
        print('warning: projection did not converge')

    r = rmag*dc

    return MeshVertex(r, dpdx, dpdy, dpdz, *args[:-1])

def discretize_wd_style(N, q, F, d, Phi):
    """
    TODO: add documentation

    New implementation. I'll make this work first, then document.
    """

    DEBUG = False

    Ts = []

    potential = 'BinaryRoche'
    r0 = -project_onto_potential(np.array((-0.02, 0.0, 0.0)), potential, q, d, F, Phi).r[0]

    # The following is a hack that needs to go!
    pot_name = potential
    dpdx = globals()['d%sdx'%(pot_name)]
    dpdy = globals()['d%sdy'%(pot_name)]
    dpdz = globals()['d%sdz'%(pot_name)]

    if DEBUG:
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        import matplotlib.patches as patches

        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.set_xlim(-0.3, 0.3) # -1.6 1.6
        ax1.set_ylim(-0.3, 0.3)
        ax2.set_xlim(-0.3, 0.3)
        ax2.set_ylim(-0.3, 0.3)
        ax3.set_xlim(-0.3, 0.3)
        ax3.set_ylim(-0.3, 0.3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        ax3.set_xlabel('y')
        ax3.set_ylabel('z')

    # Rectangle centers:
    theta = np.array([np.pi/2*(k-0.5)/N for k in range(1, N+2)])
    phi = np.array([[np.pi*(l-0.5)/Mk for l in range(1, Mk+1)] for Mk in np.array(1 + 1.3*N*np.sin(theta), dtype=int)])

    for t in range(len(theta)-1):
        dtheta = theta[t+1]-theta[t]
        for i in range(len(phi[t])):
            dphi = phi[t][1]-phi[t][0]

            # Project the vertex onto the potential; this will be our center point:
            rc = np.array((r0*sin(theta[t])*cos(phi[t][i]), r0*sin(theta[t])*sin(phi[t][i]), r0*cos(theta[t])))
            vc = project_onto_potential(rc, potential, d, q, F, Phi).r

            # Next we need to find the tangential plane, which we'll get by finding the normal,
            # which is the negative of the gradient:
            nc = np.array((-dpdx(vc, d, q, F), -dpdy(vc, d, q, F), -dpdz(vc, d, q, F)))

            # Then we need to find the intercontext of +/-dtheta/dphi-deflected
            # radius vectors with the tangential plane. We do that by solving
            #
            #   d = [(p0 - l0) \dot n] / (l \dot n),
            #
            # where p0 and l0 are reference points on the plane and on the line,
            # respectively, n is the normal vector, and l in the line direction
            # vector. For convenience l0 can be set to 0, and p0 is just vc. d
            # then measures the distance from the origin along l.

            l1 = np.array((sin(theta[t]-dtheta/2)*cos(phi[t][i]-dphi/2), sin(theta[t]-dtheta/2)*sin(phi[t][i]-dphi/2), cos(theta[t]-dtheta/2)))
            l2 = np.array((sin(theta[t]-dtheta/2)*cos(phi[t][i]+dphi/2), sin(theta[t]-dtheta/2)*sin(phi[t][i]+dphi/2), cos(theta[t]-dtheta/2)))
            l3 = np.array((sin(theta[t]+dtheta/2)*cos(phi[t][i]+dphi/2), sin(theta[t]+dtheta/2)*sin(phi[t][i]+dphi/2), cos(theta[t]+dtheta/2)))
            l4 = np.array((sin(theta[t]+dtheta/2)*cos(phi[t][i]-dphi/2), sin(theta[t]+dtheta/2)*sin(phi[t][i]-dphi/2), cos(theta[t]+dtheta/2)))

            r1 = np.dot(vc, nc) / np.dot(l1, nc) * l1
            r2 = np.dot(vc, nc) / np.dot(l2, nc) * l2
            r3 = np.dot(vc, nc) / np.dot(l3, nc) * l3
            r4 = np.dot(vc, nc) / np.dot(l4, nc) * l4

            # This sorts out the vertices, now we need to fudge the surface
            # area. WD does not take curvature of the equipotential at vc
            # into account, so the surface area computed from these vertex-
            # delimited surfaces will generally be different from what WD
            # computes. Thus, we compute the surface area the same way WD
            # does it and assign it to each element even though that isn't
            # quite its area:
            #
            #   dsigma = r^2 sin(theta)/cos(gamma) dtheta dphi,
            #
            # where gamma is the angle between l and n.

            cosgamma = np.dot(vc, nc)/np.sqrt(np.dot(vc, vc))/np.sqrt(np.dot(nc, nc))
            dsigma = np.dot(vc, vc)*np.sin(theta[t])/cosgamma*dtheta*dphi

            if DEBUG:
                fc = 'orange'

                verts = [(r1[0], r1[1]), (r2[0], r2[1]), (r3[0], r3[1]), (r4[0], r4[1]), (r1[0], r1[1])]
                codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor=fc, lw=2)
                ax1.add_patch(patch)

                verts = [(r1[0], r1[2]), (r2[0], r2[2]), (r3[0], r3[2]), (r4[0], r4[2]), (r1[0], r1[2])]
                codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor=fc, lw=2)
                ax2.add_patch(patch)

                verts = [(r1[1], r1[2]), (r2[1], r2[2]), (r3[1], r3[2]), (r4[1], r4[2]), (r1[1], r1[2])]
                codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor=fc, lw=2)
                ax3.add_patch(patch)

            # Ts.append(np.array((vc[0], vc[1], vc[2], dsigma/2, r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], r3[0], r3[1], r3[2], nc[0], nc[1], nc[2])))
            # Ts.append(np.array((vc[0], vc[1], vc[2], dsigma/2, r3[0], r3[1], r3[2], r4[0], r4[1], r4[2], r1[0], r1[1], r1[2], nc[0], nc[1], nc[2])))

            # # Instead of recomputing all quantities, just reflect over the y- and z-directions.
            # Ts.append(np.array((vc[0], -vc[1],  vc[2], dsigma/2, r1[0], -r1[1],  r1[2], r2[0], -r2[1],  r2[2], r3[0], -r3[1],  r3[2], nc[0], -nc[1], nc[2])))
            # Ts.append(np.array((vc[0], -vc[1],  vc[2], dsigma/2, r3[0], -r3[1],  r3[2], r4[0], -r4[1],  r4[2], r1[0], -r1[1],  r1[2], nc[0], -nc[1], nc[2])))

            # Ts.append(np.array((vc[0],  vc[1], -vc[2], dsigma/2, r1[0],  r1[1], -r1[2], r2[0],  r2[1], -r2[2], r3[0],  r3[1], -r3[2], nc[0],  nc[1], -nc[2])))
            # Ts.append(np.array((vc[0],  vc[1], -vc[2], dsigma/2, r3[0],  r3[1], -r3[2], r4[0],  r4[1], -r4[2], r1[0],  r1[1], -r1[2], nc[0],  nc[1], -nc[2])))

            # Ts.append(np.array((vc[0], -vc[1], -vc[2], dsigma/2, r1[0], -r1[1], -r1[2], r2[0], -r2[1], -r2[2], r3[0], -r3[1], -r3[2], nc[0], -nc[1], -nc[2])))
            # Ts.append(np.array((vc[0], -vc[1], -vc[2], dsigma/2, r3[0], -r3[1], -r3[2], r4[0], -r4[1], -r4[2], r1[0], -r1[1], -r1[2], nc[0], -nc[1], -nc[2])))

            # FOR TESTING - report theta/phi for each triangle
            # uncomment the above original version eventually
            Ts.append(np.array((vc[0], vc[1], vc[2], dsigma/2, r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], r3[0], r3[1], r3[2], nc[0], nc[1], nc[2], theta[t], phi[t][0])))
            Ts.append(np.array((vc[0], vc[1], vc[2], dsigma/2, r3[0], r3[1], r3[2], r4[0], r4[1], r4[2], r1[0], r1[1], r1[2], nc[0], nc[1], nc[2], theta[t], phi[t][0])))

            # Instead of recomputing all quantities, just reflect over the y- and z-directions.
            Ts.append(np.array((vc[0], -vc[1],  vc[2], dsigma/2, r1[0], -r1[1],  r1[2], r2[0], -r2[1],  r2[2], r3[0], -r3[1],  r3[2], nc[0], -nc[1], nc[2], theta[t], -phi[t][0])))
            Ts.append(np.array((vc[0], -vc[1],  vc[2], dsigma/2, r3[0], -r3[1],  r3[2], r4[0], -r4[1],  r4[2], r1[0], -r1[1],  r1[2], nc[0], -nc[1], nc[2], theta[t], -phi[t][0])))

            Ts.append(np.array((vc[0],  vc[1], -vc[2], dsigma/2, r1[0],  r1[1], -r1[2], r2[0],  r2[1], -r2[2], r3[0],  r3[1], -r3[2], nc[0],  nc[1], -nc[2], np.pi-theta[t], phi[t][0])))
            Ts.append(np.array((vc[0],  vc[1], -vc[2], dsigma/2, r3[0],  r3[1], -r3[2], r4[0],  r4[1], -r4[2], r1[0],  r1[1], -r1[2], nc[0],  nc[1], -nc[2], np.pi-theta[t], phi[t][0])))

            Ts.append(np.array((vc[0], -vc[1], -vc[2], dsigma/2, r1[0], -r1[1], -r1[2], r2[0], -r2[1], -r2[2], r3[0], -r3[1], -r3[2], nc[0], -nc[1], -nc[2], np.pi-theta[t], -phi[t][0])))
            Ts.append(np.array((vc[0], -vc[1], -vc[2], dsigma/2, r3[0], -r3[1], -r3[2], r4[0], -r4[1], -r4[2], r1[0], -r1[1], -r1[2], nc[0], -nc[1], -nc[2], np.pi-theta[t], -phi[t][0])))

    if DEBUG:
        plt.show()

    # Assemble a mesh table:
    table = np.array(Ts)
    return table