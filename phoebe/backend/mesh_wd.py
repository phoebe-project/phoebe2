import numpy as np

from math import sqrt, sin, cos, acos, atan2, trunc, pi
from .oc_geometry import nekmin, wd_mesh_fill, wd_recompute_neck
import libphoebe
import os
from phoebe.distortions.roche import *
# from scipy.spatial import KDTree

import logging
logger = logging.getLogger("MESH_WD")
logger.addHandler(logging.NullHandler())


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
        logger.warning('projection did not converge')

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
    r0 = libphoebe.roche_pole(q, F, d, Phi)

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
            # respectively, n is the normal vector, and l is the line direction
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
            #   dsigma = || r^2 sin(theta)/cos(gamma) dtheta dphi ||,
            #
            # where gamma is the angle between l and n.

            cosgamma = np.dot(vc, nc)/np.sqrt(np.dot(vc, vc))/np.sqrt(np.dot(nc, nc))
            dsigma = np.abs(np.dot(vc, vc)*np.sin(theta[t])/cosgamma*dtheta*dphi)

            # Temporary addition: triangle areas: ######################
            side1 = sqrt((r1[0]-r2[0])**2 + (r1[1]-r2[1])**2 + (r1[2]-r2[2])**2)
            side2 = sqrt((r1[0]-r3[0])**2 + (r1[1]-r3[1])**2 + (r1[2]-r3[2])**2)
            side3 = sqrt((r2[0]-r3[0])**2 + (r2[1]-r3[1])**2 + (r2[2]-r3[2])**2)
            s = 0.5*(side1 + side2 + side3)

            dsigma_t_sq = s*(s-side1)*(s-side2)*(s-side3)
            dsigma_t = sqrt(dsigma_t_sq) if dsigma_t_sq > 0 else 0.0
            ############################################################

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
            Ts.append(np.array((vc[0], vc[1], vc[2], dsigma/2, r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], r3[0], r3[1], r3[2], nc[0], nc[1], nc[2], theta[t], phi[t][0], dsigma_t)))
            Ts.append(np.array((vc[0], vc[1], vc[2], dsigma/2, r3[0], r3[1], r3[2], r4[0], r4[1], r4[2], r1[0], r1[1], r1[2], nc[0], nc[1], nc[2], theta[t], phi[t][0], dsigma_t)))

            # Instead of recomputing all quantities, just reflect over the y- and z-directions.
            Ts.append(np.array((vc[0], -vc[1],  vc[2], dsigma/2, r1[0], -r1[1],  r1[2], r2[0], -r2[1],  r2[2], r3[0], -r3[1],  r3[2], nc[0], -nc[1], nc[2], theta[t], -phi[t][0], dsigma_t)))
            Ts.append(np.array((vc[0], -vc[1],  vc[2], dsigma/2, r3[0], -r3[1],  r3[2], r4[0], -r4[1],  r4[2], r1[0], -r1[1],  r1[2], nc[0], -nc[1], nc[2], theta[t], -phi[t][0], dsigma_t)))

            Ts.append(np.array((vc[0],  vc[1], -vc[2], dsigma/2, r1[0],  r1[1], -r1[2], r2[0],  r2[1], -r2[2], r3[0],  r3[1], -r3[2], nc[0],  nc[1], -nc[2], np.pi-theta[t], phi[t][0], dsigma_t)))
            Ts.append(np.array((vc[0],  vc[1], -vc[2], dsigma/2, r3[0],  r3[1], -r3[2], r4[0],  r4[1], -r4[2], r1[0],  r1[1], -r1[2], nc[0],  nc[1], -nc[2], np.pi-theta[t], phi[t][0], dsigma_t)))

            Ts.append(np.array((vc[0], -vc[1], -vc[2], dsigma/2, r1[0], -r1[1], -r1[2], r2[0], -r2[1], -r2[2], r3[0], -r3[1], -r3[2], nc[0], -nc[1], -nc[2], np.pi-theta[t], -phi[t][0], dsigma_t)))
            Ts.append(np.array((vc[0], -vc[1], -vc[2], dsigma/2, r3[0], -r3[1], -r3[2], r4[0], -r4[1], -r4[2], r1[0], -r1[1], -r1[2], nc[0], -nc[1], -nc[2], np.pi-theta[t], -phi[t][0], dsigma_t)))

    if DEBUG:
        plt.show()

    # Assemble a mesh table:
    table = np.array(Ts)
    return table

def discretize_wd_style_oc(N, q, F, d, Phi,recompute_neck=True):

    Ts = []
    Tstart = [0,]

    # for testing of where things start to diverge:
    breaks0, breaks1 = [], []
    vcs0, vcs1 = [], []

    potential = 'BinaryRoche'
    q_1, Phi_1 = q, Phi
    q_2, Phi_2 = 1./q, Phi/q + 0.5*(q-1)/q

    xmin1, z1 = nekmin(Phi_1,q_1,0.5,0.05,0.05)
    # xmin1 = (xminz1+xminy1)/2.

    xmin2 = d - xmin1
    #xminz2, xminy2, y2, z2 = nekmin(Phi_2,q_2,0.5,0.05,0.05)

    # first, find closest neck points to mark them for fractional area computation
    for obj, q, Phi, xmin in zip([0,1],[q_1,q_2],[Phi_1,Phi_2],[xmin1,xmin2]):

        r0 = libphoebe.roche_pole(q, F, d, Phi)
        # The following is a hack that needs to go!
        pot_name = potential
        dpdx = globals()['d%sdx'%(pot_name)]
        dpdy = globals()['d%sdy'%(pot_name)]
        dpdz = globals()['d%sdz'%(pot_name)]

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

                if (vc[0] < 0. and vc[0] > -1.) or abs(vc[0]) <= xmin:
                    # do this for separate components cause they may have different diverging strips
                    if obj==0 and abs(vc[1])>=1e-16:
                        vcs0.append(np.array([vc[0],vc[1],vc[2],theta[t]]))
                    elif obj==1 and abs(vc[1])>=1e-16:
                        vcs1.append(np.array([vc[0],vc[1],vc[2],theta[t]]))
                else:
                    if obj==0 and abs(vc[1])>=1e-16:
                        breaks0.append(theta[t])
                    elif obj==1 and abs(vc[1])>=1e-16:
                        breaks1.append(theta[t])

    vcs0,vcs1 = np.array(vcs0), np.array(vcs1)
    #breaks0, breaks1 = np.array(breaks0), np.array(breaks1)
    thetas0, thetas1 = set(breaks0), set(breaks1)

    vcs_breaks0 = []
    vcs_breaks1 = []
    # go through strips of thetas where things diverge and mark last created center

    for theta in thetas0:
        vcs_strip = vcs0[vcs0[:,-1]==theta]
        vcs_breaks0.append([vcs_strip[0][0],vcs_strip[0][1],vcs_strip[0][2]])

    for theta in thetas1:
        vcs_strip = vcs1[vcs1[:,-1]==theta]
        vcs_breaks1.append([vcs_strip[0][0],vcs_strip[0][1],vcs_strip[0][2]])

    vcs_breaks0 = np.array(vcs_breaks0)
    vcs_breaks1 = np.array(vcs_breaks1)

    for obj, q, Phi, xmin, vcs_breaks in zip([0,1],[q_1,q_2],[Phi_1,Phi_2],[xmin1,xmin2],[vcs_breaks0,vcs_breaks1]):

        r0 = libphoebe.roche_pole(q, F, d, Phi)
        # The following is a hack that needs to go!
        pot_name = potential
        dpdx = globals()['d%sdx'%(pot_name)]
        dpdy = globals()['d%sdy'%(pot_name)]
        dpdz = globals()['d%sdz'%(pot_name)]

        # Rectangle centers:
        theta = np.array([np.pi/2*(k-0.5)/N for k in range(1, N+2)])
        phi = np.array([[np.pi*(l-0.5)/Mk for l in range(1, Mk+1)] for Mk in np.array(1 + 1.3*N*np.sin(theta), dtype=int)])


        for t in range(len(theta)-1):
            dtheta = theta[t+1]-theta[t]
            for i in range(len(phi[t])):
                dphi = phi[t][1]-phi[t][0]

                # Project the vertex onto the potential; this will be our center point:
                # print "projecting center"
                rc = np.array((r0*sin(theta[t])*cos(phi[t][i]), r0*sin(theta[t])*sin(phi[t][i]), r0*cos(theta[t])))
                vc = project_onto_potential(rc, potential, d, q, F, Phi).r

                if ((vc[0] < 0. and vc[0] > -1.) or abs(vc[0]) <= xmin) and abs(vc[1])>=1e-16:

                    # Next we need to find the tangential plane, which we'll get by finding the normal,
                    # which is the negative of the gradient:
                    nc = np.array((-dpdx(vc, d, q, F), -dpdy(vc, d, q, F), -dpdz(vc, d, q, F)))

                    # Then we need to find the intercontext of +/-dtheta/dphi-deflected
                    # radius vectors with the tangential plane. We do that by solving
                    #
                    #   d = [(p0 - l0) \dot n] / (l \dot n),
                    #
                    # where p0 and l0 are reference points on the plane and on the line,
                    # respectively, n is the normal vector, and l is the line direction
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
                    #   dsigma = || r^2 sin(theta)/cos(gamma) dtheta dphi ||,
                    #
                    # where gamma is the angle between l and n.

                    cosgamma = np.dot(vc, nc)/np.sqrt(np.dot(vc, vc))/np.sqrt(np.dot(nc, nc))
                    dsigma = np.abs(np.dot(vc, vc)*np.sin(theta[t])/cosgamma*dtheta*dphi)

                    # Temporary addition: triangle areas: ######################
                    side1 = sqrt((r1[0]-r2[0])**2 + (r1[1]-r2[1])**2 + (r1[2]-r2[2])**2)
                    side2 = sqrt((r1[0]-r3[0])**2 + (r1[1]-r3[1])**2 + (r1[2]-r3[2])**2)
                    side3 = sqrt((r2[0]-r3[0])**2 + (r2[1]-r3[1])**2 + (r2[2]-r3[2])**2)
                    s = 0.5*(side1 + side2 + side3)

                    dsigma_t_sq = s*(s-side1)*(s-side2)*(s-side3)
                    dsigma_t = sqrt(dsigma_t_sq) if dsigma_t_sq > 0 else 0.0
                    ############################################################

                    #if abs(r1[0]) <= xmin and abs(r2[0]) <= xmin and abs(r3[0]) <= xmin and abs(r4[0]) <= xmin:

                    # check whether the trapezoids cross the neck and if so put them back into their place
                    # do the same for end traezoids also

                    if (np.array([abs(r1[0]),abs(r2[0]),abs(r3[0]),abs(r4[0])])>xmin).any() or vc in vcs_breaks:
                        r1,r2,r3,r4,dsigma_t_new = wd_mesh_fill(xmin,y1,z1,r1,r2,r3,r4)
                        if dsigma_t != 0:
                            frac_area = dsigma_t_new/dsigma_t
                            dsigma = dsigma*frac_area
                            dsigma_t = dsigma_t*frac_area
                        else:
                            dsigma_t = dsigma_t_new
                            dsigma = 2*dsigma_t_new

                        if recompute_neck:
                            #vc,nc,r1,r2,r3,r4,dsigma,dsigma_t,phi_half,dphi_new=wd_recompute_neck(r0,xmin,y1,z1,r1,r2,r3,r4,theta[t],dtheta,potential,d,q,F,Phi)
                            disgma = wd_recompute_neck(r0,xmin,y1,z1,r1,r2,r3,r4,theta[t],dtheta,potential,d,q,F,Phi)
                            #phi[t][i],dphi=phi_half,dphi_new

                        # If all of the vertices are on one side of the minimum, keep trapezoids as they are

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


                    # Temporary addition: vertex normals: ######################
                    # print "projecting vertices "
                    v1 = project_onto_potential(r1, potential, d, q, F, Phi).r
                    v2 = project_onto_potential(r2, potential, d, q, F, Phi).r
                    v3 = project_onto_potential(r3, potential, d, q, F, Phi).r
                    v4 = project_onto_potential(r4, potential, d, q, F, Phi).r

                    n1 = np.array((-dpdx(v1, d, q, F), -dpdy(v1, d, q, F), -dpdz(v1, d, q, F)))
                    n2 = np.array((-dpdx(v2, d, q, F), -dpdy(v2, d, q, F), -dpdz(v2, d, q, F)))
                    n3 = np.array((-dpdx(v3, d, q, F), -dpdy(v3, d, q, F), -dpdz(v3, d, q, F)))
                    n4 = np.array((-dpdx(v4, d, q, F), -dpdy(v4, d, q, F), -dpdz(v4, d, q, F)))

                    ############################################################

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
                    if obj==0:
                        Ts.append(np.array((vc[0], vc[1], vc[2], dsigma/2, r3[0], r3[1], r3[2], r4[0], r4[1], r4[2], r1[0], r1[1], r1[2], nc[0], nc[1], nc[2], theta[t], phi[t][0], dsigma_t, n3[0], n3[1], n3[2], n4[0], n4[1], n4[2], n1[0], n1[1], n1[2])))
                        Ts.append(np.array((vc[0], vc[1], vc[2], dsigma/2, r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], r3[0], r3[1], r3[2], nc[0], nc[1], nc[2], theta[t], phi[t][0], dsigma_t, n1[0], n1[1], n1[2], n2[0], n2[1], n2[2], n3[0], n3[1], n3[2])))

                    else:
                        Ts.append(np.array((-vc[0]+d, vc[1], vc[2], dsigma/2, -r3[0]+d, r3[1], r3[2], -r4[0]+d, r4[1], r4[2], -r1[0]+d, r1[1], r1[2], -nc[0], nc[1], nc[2], theta[t], phi[t][0], dsigma_t, -n3[0], n3[1], n3[2], -n4[0], n4[1], n4[2], -n1[0], n1[1], n1[2])))
                        Ts.append(np.array((-vc[0]+d, vc[1], vc[2], dsigma/2, -r1[0]+d, r1[1], r1[2], -r2[0]+d, r2[1], r2[2], -r3[0]+d, r3[1], r3[2], -nc[0], nc[1], nc[2], theta[t], phi[t][0], dsigma_t, -n1[0], n1[1], n1[2], -n2[0], n2[1], n2[2], -n3[0], n3[1], n3[2])))

        for T in reversed(Ts[Tstart[-1]:]):
            Ts.append(np.array((T[0], -T[1],  T[2], T[3], T[4], -T[5],  T[6], T[7], -T[8],  T[9], T[10], -T[11],  T[12], T[13], -T[14], T[15], T[16], -T[17], T[18], T[19], -T[20],  T[21], T[22], -T[23],  T[24], T[25], -T[26],  T[27])))
        Tstart.append(len(Ts))

    for i in range(len(Tstart)-1):
        for T in Ts[Tstart[-2]:Tstart[-1]]:
            Ts.append(np.array((T[0],  T[1], -T[2], T[3], T[4],  T[5], -T[6], T[7],  T[8], -T[9], T[10],  T[11], -T[12], T[13],  T[14], -T[15], np.pi-T[16], T[17], T[18], T[19],  T[20], -T[21], T[22],  T[23], -T[24], T[25],  T[26], -T[27])))
        Tstart.pop(-1)

                    # if obj == 0:
                    #     Ts.append(np.array((vc[0], vc[1], vc[2], dsigma/2, r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], r3[0], r3[1], r3[2], nc[0], nc[1], nc[2], theta[t], phi[t][0], dsigma_t)))
                    #     Ts.append(np.array((vc[0], vc[1], vc[2], dsigma/2, r3[0], r3[1], r3[2], r4[0], r4[1], r4[2], r1[0], r1[1], r1[2], nc[0], nc[1], nc[2], theta[t], phi[t][0], dsigma_t)))

                    #     # Instead of recomputing all quantities, just reflect over the y- and z-directions.
                    #     Ts.append(np.array((vc[0], -vc[1],  vc[2], dsigma/2, r1[0], -r1[1],  r1[2], r2[0], -r2[1],  r2[2], r3[0], -r3[1],  r3[2], nc[0], -nc[1], nc[2], theta[t], -phi[t][0], dsigma_t)))
                    #     Ts.append(np.array((vc[0], -vc[1],  vc[2], dsigma/2, r3[0], -r3[1],  r3[2], r4[0], -r4[1],  r4[2], r1[0], -r1[1],  r1[2], nc[0], -nc[1], nc[2], theta[t], -phi[t][0], dsigma_t)))

                    #     Ts.append(np.array((vc[0],  vc[1], -vc[2], dsigma/2, r1[0],  r1[1], -r1[2], r2[0],  r2[1], -r2[2], r3[0],  r3[1], -r3[2], nc[0],  nc[1], -nc[2], np.pi-theta[t], phi[t][0], dsigma_t)))
                    #     Ts.append(np.array((vc[0],  vc[1], -vc[2], dsigma/2, r3[0],  r3[1], -r3[2], r4[0],  r4[1], -r4[2], r1[0],  r1[1], -r1[2], nc[0],  nc[1], -nc[2], np.pi-theta[t], phi[t][0], dsigma_t)))

                    #     Ts.append(np.array((vc[0], -vc[1], -vc[2], dsigma/2, r1[0], -r1[1], -r1[2], r2[0], -r2[1], -r2[2], r3[0], -r3[1], -r3[2], nc[0], -nc[1], -nc[2], np.pi-theta[t], -phi[t][0], dsigma_t)))
                    #     Ts.append(np.array((vc[0], -vc[1], -vc[2], dsigma/2, r3[0], -r3[1], -r3[2], r4[0], -r4[1], -r4[2], r1[0], -r1[1], -r1[2], nc[0], -nc[1], -nc[2], np.pi-theta[t], -phi[t][0], dsigma_t)))
                    # else:
                    #     Ts.append(np.array((-vc[0]+d, vc[1], vc[2], dsigma/2, -r1[0]+d, r1[1], r1[2], -r2[0]+d, r2[1], r2[2], -r3[0]+d, r3[1], r3[2], -nc[0], nc[1], nc[2], theta[t], phi[t][0], dsigma_t)))
                    #     Ts.append(np.array((-vc[0]+d, vc[1], vc[2], dsigma/2, -r3[0]+d, r3[1], r3[2], -r4[0]+d, r4[1], r4[2], -r1[0]+d, r1[1], r1[2], -nc[0], nc[1], nc[2], theta[t], phi[t][0], dsigma_t)))

                    #     # Instead of recomputing all quantities, just reflect over the y- and z-directions.
                    #     Ts.append(np.array((-vc[0]+d, -vc[1],  vc[2], dsigma/2, -r1[0]+d, -r1[1],  r1[2], -r2[0]+d, -r2[1],  r2[2], -r3[0]+d, -r3[1],  r3[2], -nc[0], -nc[1], nc[2], theta[t], -phi[t][0], dsigma_t)))
                    #     Ts.append(np.array((-vc[0]+d, -vc[1],  vc[2], dsigma/2, -r3[0]+d, -r3[1],  r3[2], -r4[0]+d, -r4[1],  r4[2], -r1[0]+d, -r1[1],  r1[2], -nc[0], -nc[1], nc[2], theta[t], -phi[t][0], dsigma_t)))

                    #     Ts.append(np.array((-vc[0]+d,  vc[1], -vc[2], dsigma/2, -r1[0]+d,  r1[1], -r1[2], -r2[0]+d,  r2[1], -r2[2], -r3[0]+d,  r3[1], -r3[2], -nc[0],  nc[1], -nc[2], np.pi-theta[t], phi[t][0], dsigma_t)))
                    #     Ts.append(np.array((-vc[0]+d,  vc[1], -vc[2], dsigma/2, -r3[0]+d,  r3[1], -r3[2], -r4[0]+d,  r4[1], -r4[2], -r1[0]+d,  r1[1], -r1[2], -nc[0],  nc[1], -nc[2], np.pi-theta[t], phi[t][0], dsigma_t)))

                    #     Ts.append(np.array((-vc[0]+d, -vc[1], -vc[2], dsigma/2, -r1[0]+d, -r1[1], -r1[2], -r2[0]+d, -r2[1], -r2[2], -r3[0]+d, -r3[1], -r3[2], -nc[0], -nc[1], -nc[2], np.pi-theta[t], -phi[t][0], dsigma_t)))
                    #     Ts.append(np.array((-vc[0]+d, -vc[1], -vc[2], dsigma/2, -r3[0]+d, -r3[1], -r3[2], -r4[0]+d, -r4[1], -r4[2], -r1[0]+d, -r1[1], -r1[2], -nc[0], -nc[1], -nc[2], np.pi-theta[t], -phi[t][0], dsigma_t)))


    # Assemble a mesh table:
    table = np.array(Ts)
    return table
