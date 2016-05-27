import numpy as np

import numpy.lib.recfunctions as recfunctions
import matplotlib.mlab as mlab

from scipy.optimize import newton
from math import sqrt, sin, cos, acos, atan2, trunc, pi
import os
from phoebe.utils import coordinates, fgeometry
from phoebe.algorithms import cmarching
from phoebe.atmospheres import limbdark, passbands
from phoebe.dynamics import ctrans
from phoebe.distortions import roche
from phoebe.backend import eclipse, subdivision
import copy


from phoebe2 import u
from phoebe2 import c as constants # TODO: change these to c

import logging
logger = logging.getLogger("MESH")
logger.addHandler(logging.NullHandler())

default_polar_dir = np.array([0,0,-1.0])
default_los_dir = np.array([0,0,+1.0])
default_zeros = np.array([0,0,0.0])

_basedir = os.path.dirname(os.path.abspath(__file__))
_pbdir = os.path.abspath(os.path.join(_basedir, '..', 'atmospheres', 'tables', 'passbands'))


def _value(obj):
    """
    make sure to get a float
    """
    # TODO: this is ugly and makes everything ugly
    # can we handle this with a clean decorator or just requiring that only floats be passed??
    if hasattr(obj, 'value'):
        return obj.value
    elif isinstance(obj, np.ndarray):
        return np.array([o.value for o in obj])
    elif hasattr(obj, '__iter__'):
        return [o.value for o in obj]
    return obj

def compute_volume(sizes, centers, normals):
    """
    Compute the numerical volume of a convex

    :parameter array sizes: array of sizes of triangles
    :parameter array centers: array of centers of triangles (x,y,z)
    :parameter array normals: array of normals of triangles (will normalize if not already)
    """
    # the volume of a slanted triangular cone is A_triangle * (r_vec dot norm_vec) / 3.

    # TODO: implement normalizing normals into meshing routines (or at least have them supply normal_mags to the mesh)

    normal_mags = np.sqrt((normals**2).sum(axis=1))
    return np.sum(sizes*((centers*normals).sum(axis=1)/normal_mags)/3)



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

def discretize_wd_style(N=30, potential='BinaryRoche', *args):
    """
    TODO: add documentation

    New implementation. I'll make this work first, then document.
    """

    DEBUG = False

    Ts = []

    r0 = -project_onto_potential(np.array((-0.02, 0.0, 0.0)), potential, *args).r[0]

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
            vc = project_onto_potential(rc, potential, *args).r

            # Next we need to find the tangential plane, which we'll get by finding the normal,
            # which is the negative of the gradient:
            nc = np.array((-dpdx(vc, *args[:-1]), -dpdy(vc, *args[:-1]), -dpdz(vc, *args[:-1])))

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

            Ts.append(np.array((vc[0], vc[1], vc[2], dsigma/2, r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], r3[0], r3[1], r3[2], nc[0], nc[1], nc[2])))
            Ts.append(np.array((vc[0], vc[1], vc[2], dsigma/2, r3[0], r3[1], r3[2], r4[0], r4[1], r4[2], r1[0], r1[1], r1[2], nc[0], nc[1], nc[2])))

            # Instead of recomputing all quantities, just reflect over the y- and z-directions.
            Ts.append(np.array((vc[0], -vc[1],  vc[2], dsigma/2, r1[0], -r1[1],  r1[2], r2[0], -r2[1],  r2[2], r3[0], -r3[1],  r3[2], nc[0], -nc[1], nc[2])))
            Ts.append(np.array((vc[0], -vc[1],  vc[2], dsigma/2, r3[0], -r3[1],  r3[2], r4[0], -r4[1],  r4[2], r1[0], -r1[1],  r1[2], nc[0], -nc[1], nc[2])))

            Ts.append(np.array((vc[0],  vc[1], -vc[2], dsigma/2, r1[0],  r1[1], -r1[2], r2[0],  r2[1], -r2[2], r3[0],  r3[1], -r3[2], nc[0],  nc[1], -nc[2])))
            Ts.append(np.array((vc[0],  vc[1], -vc[2], dsigma/2, r3[0],  r3[1], -r3[2], r4[0],  r4[1], -r4[2], r1[0],  r1[1], -r1[2], nc[0],  nc[1], -nc[2])))

            Ts.append(np.array((vc[0], -vc[1], -vc[2], dsigma/2, r1[0], -r1[1], -r1[2], r2[0], -r2[1], -r2[2], r3[0], -r3[1], -r3[2], nc[0], -nc[1], -nc[2])))
            Ts.append(np.array((vc[0], -vc[1], -vc[2], dsigma/2, r3[0], -r3[1], -r3[2], r4[0], -r4[1], -r4[2], r1[0], -r1[1], -r1[2], nc[0], -nc[1], -nc[2])))

    if DEBUG:
        plt.show()

    # Assemble a mesh table:
    table = np.array(Ts)
    return table


##############################################################################################
##############################################################################################
##############################################################################################

class Meshes(object):
    """
    This is just a class to allow easy access to meshes at the system-level.
    This allows eclipse detection and subdivision to work at the system-level in
    an efficient and clean way.

    In effect this has the same structure as the system, but allows passing only
    the meshes and dropping all the Body methods/attributes.
    """
    def __init__(self, items):
        """
        TODO: add documentation

        :parameter list items:
        """
        self._dict = {component: body.mesh for component, body in items.items()}
        #self._component_by_no = {body.comp_no: component for component, body in items.items()}
        self._components = items.keys()

    def items(self):
        """
        TODO: add documentation
        """
        return self._dict.items()

    def keys(self):
        """
        TODO: add documentation
        """
        return self._dict.keys()

    def values(self):
        """
        TODO: add documentation
        """
        return self._dict.values()

    def __getitem__(self, key):
        """
        """
        return self._dict[key]

    def __setitem__(self, key, value):
        """
        """
        return self.update_columns(key, value)

    def component_by_no(self, comp_no):
        """
        TODO: add documentation
        """
        #return self._component_by_no[comp_no]
        return self._components[comp_no-1]

    def update_columns(self, field, value_dict, inds=None):
        """
        update the columns of all meshes

        :parameter str field: name of the mesh columnname
        :parameter value_dict: dictionary with component as keys and new
            data as values. If value_dict is not a dictionary,
            it will be applied to all components
        :type value_dict: dict or value (array or float)
        """
        if not isinstance(value_dict, dict):
            value_dict = {comp_no: value_dict for comp_no in self._dict.keys()}


        for comp, value in value_dict.items():
            #print "***", comp, field, inds, value
            if inds:
                self._dict[comp][field][inds] = value
            else:
                self._dict[comp][field] = value

    def get_column(self, field, components=None):
        """
        TODO: add documentation

        return a dictionary for a single column, with component as keys and the
        column array as values

        :parameter str field: name of the mesh columnname
        :parameter components:
        """
        if components:
            if isinstance(components, str):
                components = [components]
            return {c: self._dict[c][field] for c in components}
        else:
            return {c: m[field] for c, m in self._dict.items()}

    def get_column_flat(self, field, components=None):
        """
        TODO: add documentation

        return a single merged value (hstacked) from all meshes

        :parameter str field: name of the mesh columnname
        :parameter components:
        """
        if components:
            if isinstance(components, str):
                components = [components]
            return np.hstack([self._dict[c][field] for c in components])
        else:
            return np.hstack([m[field] for m in self._dict.values()])

    def replace_elements(self, inds, new_submesh, component):
        """
        TODO: add documentation
        """
        self._dict[component] = np.hstack([self._dict[component][~inds], new_submesh])


class System(object):
    def __init__(self, bodies_dict, eclipse_alg='graham', subdiv_alg='edge', subdiv_num=3, dynamics_method='keplerian'):
        """
        :parameter dict bodies_dict: dictionary of component names and Bodies (or subclass of Body)
        """
        self._bodies = bodies_dict
        self.eclipse_alg = eclipse_alg
        self.subdiv_alg = subdiv_alg
        self.subdiv_num = subdiv_num
        self.dynamics_method = dynamics_method

        return

    @classmethod
    def from_bundle(cls, b, compute=None, datasets=[], **kwargs):
        """
        Build a system from the :class:`phoebe.frontend.bundle.Bundle` and its
        hierarchy.

        :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
        :parameter str compute: name of the computeoptions in the bundle
        :parameter list datasets: list of names of datasets
        :parameter **kwargs: temporary overrides for computeoptions
        :return: an instantiated :class:`System` object, including its children
            :class:`Body`s
        """

        hier = b.hierarchy

        if not len(hier.get_value()):
            raise NotImplementedError("Meshing requires a hierarchy to exist")


        # now pull general compute options
        if compute is not None:
            if isinstance(compute, str):
                compute_ps = b.get_compute(compute)
            else:
                # then hopefully compute is the parameterset
                compute_ps = compute
            eclipse_alg = compute_ps.get_value(qualifier='eclipse_alg', **kwargs)
            subdiv_alg = 'edge' #compute_ps.get_value(qualifier='subdiv_alg', **kwargs)
            subdiv_num = compute_ps.get_value(qualifier='subdiv_num', **kwargs)
            dynamics_method = compute_ps.get_value(qualifier='dynamics_method', **kwargs)
        else:
            eclipse_alg = 'graham'
            subdiv_alg = 'edge'
            subdiv_num = 3
            dynamics_method = 'keplerian'

        # NOTE: here we use globals()[Classname] because getattr doesn't work in
        # the current module - now this doesn't really make sense since we only
        # support stars, but eventually the classname could be Disk, Spot, etc
        if 'dynamics_method' in kwargs.keys():
            # already set as default above
            _dump = kwargs.pop('dynamics_method')



        #starrefs  = hier.get_stars()
        #bodies_dict = {star: globals()['Star'].from_bundle(b, star, compute, dynamics_method=dynamics_method, datasets=datasets, **kwargs) for star in starrefs}

        meshables = hier.get_meshables()
        bodies_dict = {comp: globals()[hier.get_kind_of(comp).title()].from_bundle(b, comp, compute, dynamics_method=dynamics_method, datasets=datasets, **kwargs) for comp in meshables}

        return cls(bodies_dict, eclipse_alg=eclipse_alg,
                subdiv_alg=subdiv_alg, subdiv_num=subdiv_num,
                dynamics_method=dynamics_method)

    def items(self):
        """
        TODO: add documentation
        """
        return self._bodies.items()

    def keys(self):
        """
        TODO: add documentation
        """
        return self._bodies.keys()

    def values(self):
        """
        TODO: add documentation
        """
        return self._bodies.values()

    @property
    def bodies(self):
        """
        TODO: add documentation
        """
        return self.values()

    def get_body(self, component):
        """
        TODO: add documentation
        """
        return self._bodies[component]

    @property
    def meshes(self):
        """
        TODO: add documentation
        """
        # this gives access to all methods of the Meshes class, but since everything
        # is accessed in memory (soft-copy), it will be quicker to only instantiate
        # this once.
        #
        # ie do something like this:
        #
        # meshes = self.meshes
        # meshes.update_column('visibility', visibilities)
        # meshes.update_column('somethingelse', somethingelse)
        #
        # rather than calling self.meshes repeatedly

        return Meshes(self._bodies)


    def initialize_meshes(self):
        """
        TODO: add documentation
        """
        # TODO: allow for passing theta, for now its assumed at periastron

        for starref,body in self.items():
            if not body.mesh_initialized:
                # TODO: eventually we can pass instantaneous masses and sma as kwargs if they're time dependent
                logger.info("initializing mesh for {}".format(starref))

                # This function will create the initial protomesh - centered
                # at each star's own coordinate system and not scaled by sma.
                # It will then store this mesh as a "standard" for a given theta,
                # each time can then call on one of these standards, scale using sma,
                # and reproject if necessary (for eccentricity/volume conservation)
                body.initialize_mesh()


    def update_positions(self, time, xs, ys, zs, vxs, vys, vzs, ethetas, elongans, eincls):
        """
        TODO: add documentation

        all arrays should be for the current time, but iterable over all bodies
        """
        self.xs = np.array(_value(xs))
        self.ys = np.array(_value(ys))
        self.zs = np.array(_value(zs))

        for starref,body in self.items():
            #logger.debug("updating position of mesh for {}".format(starref))
            body.update_position(time, xs, ys, zs, vxs, vys, vzs, ethetas, elongans, eincls)


    def populate_observables(self, time, methods, datasets, kwargss):
        """
        TODO: add documentation
        """

        for method, dataset, kwargs in zip(methods, datasets, kwargss):
            for starref,body in self.items():
                body.populate_observable(time, method, dataset, **kwargs)


    def handle_eclipses(self, **kwargs):
        """
        Detect the triangles at the horizon and the eclipsed triangles, handling
        any necessary subdivision.

        :parameter str eclipse_alg: name of the algorithm to use to detect
            the horizon or eclipses (defaults to the value set by computeoptions)
        :parameter str subdiv_alg: name of the algorithm to use for subdivision
            (defaults to the value set by computeoptions)
        :parameter int subdiv_num: number of subdivision iterations (defaults
            the value set by computeoptions)
        """

        eclipse_alg = kwargs.get('eclipse_alg', self.eclipse_alg)
        subdiv_alg = kwargs.get('subdiv_alg', self.subdiv_alg)
        subdiv_num = int(kwargs.get('subdiv_num', self.subdiv_num))

        # Let's first check to see if eclipses are even possible at these
        # positions.  If they are not, then we only have to do horizon
        #
        # To do that, we'll take the conservative max_r for each object
        # and their current positions, and see if the separations are larger
        # than sum of max_r
        possible_eclipse = False
        if len(self.bodies) == 1:
            # then probably an overcontact - we should eventually probably
            # deal with this differently, but for now we'll just disable
            # eclipse detection
            possible_eclipse = False
        else:
            max_rs = [body.max_r for body in self.bodies]
            for i in range(0, len(self.xs)-1):
                for j in range(i+1, len(self.xs)):
                    proj_sep_sq = sum([(c[i]-c[j])**2 for c in (self.xs,self.ys)])
                    max_sep_ecl = max_rs[i] + max_rs[j]

                    if proj_sep_sq < max_sep_ecl**2:
                        # then this pair has the potential for eclipsing triangles
                        possible_eclipse = True
                        break

        if not possible_eclipse:
            #logger.debug("eclipse not possible at this time, temporarily using eclipse_alg='only_horizon'")
            eclipse_alg = 'only_horizon'

        # meshes is an object which allows us to easily access and update columns
        # in the meshes *in memory*.  That is meshes.update_columns will propogate
        # back to the current mesh for each body.
        meshes = self.meshes

        # Reset all visibilities to be fully visible to start
        meshes.update_columns('visibility', 1.0)

        ecl_func = getattr(eclipse, eclipse_alg)

        # We need to run eclipse detection first to get the partial triangles
        # to send to subdivision
        visibilities = ecl_func(meshes, self.xs, self.ys, self.zs)
        meshes.update_columns('visibility', visibilities)

        # Now we'll handle subdivision.  If at any point there are no partially
        # subdivided triangles, then we can break the loop and we're done.  Otherwise
        # we'll continue for the number of requested iterations.
        subdiv_func = getattr(subdivision, subdiv_alg)
        for k in range(subdiv_num):
            #logger.debug("subdividing via {:s}".format(subdiv_alg))

            # TODO: try to get rid of this loop by subdividing directly on meshes?
            for component, mesh in meshes.items():
                partial = (mesh['visibility'] > 0) & (mesh['visibility'] < 1)

                if np.all(partial==False):
                    # Then we have no triangles left to subdivide
                    continue

                # TODO: what if instead of subdividing we just set the correct
                # visibility of the triangle based on what proportion of the area
                # is visible.  This would make for ugly plots though and may be
                # just as expensive to compute as this is to handle in the mesh.

                # The subdiv_func takes the mesh of ONLY the triangles that we want
                # subdivided (ie the partial triangles).  It returns a new submesh
                # of the triangles that need to replace the old trianges in the
                # mesh.  Each triangle that splits into multiple sub-triangles
                # will have any non-geometric quantities copied (ie local quantities/
                # observables are computed BEFORE subdivision and are not recomputed
                # for each individually subdivided triangle).
                new_submesh = subdiv_func(mesh[partial])


                # We now want to remove the original triangles, and add the new
                # triangles (these will surely be different lengths as we're likely
                # adding 3 or 4 times the number of triangles we remove).
                # Note: as mesh is just a member of meshes, this operation affects
                # the mesh in memory and will instantly take effect.
                meshes.replace_elements(partial, new_submesh, component)

            # TODO: at this point we only need to do eclipse detection on
            # the triangles that were just subdivided, not all the visible!
            # but of course we need to send all triangles in, since all can ECLIPSE
            # so maybe this should take a switch - of whether to consider all triangles
            # or just partial?
            visibilities = ecl_func(meshes, self.xs, self.ys, self.zs)
            meshes.update_columns('visibility', visibilities)

        if subdiv_num > 0:
            # then since adding triangles required "rebuilding" the mesh record
            # array, not everything was done in memory, so we have to push the
            # meshes back to the bodies.
            for component, body in self.items():
                body.mesh = meshes[component]

        return


    def observe(self, dataset, method, components=None, distance=1.0, l3=0.0):
        """
        TODO: add documentation

        Integrate over visible surface elements and return a dictionary of observable values

        distance (solRad)
        """

        meshes = self.meshes
        if method=='RV':
            visibility = meshes.get_column_flat('visibility', components)

            if np.all(visibility==0):
                # then no triangles are visible, so we should return nan
                #print "NO VISIBLE TRIANGLES!!!"
                return {'rv': np.nan}

            rvs = meshes.get_column_flat("rv:{}".format(dataset), components)
            intens_proj_abs = meshes.get_column_flat('intens_proj_abs:{}'.format(dataset), components)
            mu = meshes.get_column_flat('mu', components)
            area = (meshes.get_column_flat('size', components)*u.solRad**3).to(u.m**3).value

            # note that the intensities are already projected but are per unit area
            # so we need to multiply by the /projected/ area of each triangle (thus the extra mu)
            return {'rv': np.average(rvs, weights=intens_proj_abs*area*mu*visibility)}

        elif method=='LC':
            visibility = meshes.get_column_flat('visibility')

            if np.all(visibility==0):
                # then no triangles are visible, so we should return nan - probably shouldn't ever happen for lcs
                return {'flux': np.nan}

            intens_proj_rel = meshes.get_column_flat("intens_proj_rel:{}".format(dataset), components)
            mu = meshes.get_column_flat('mu', components)
            area = (meshes.get_column_flat('size', components)*u.solRad**3).to(u.m**3).value

            # intens_proj is the intensity in the direction of the observer per unit surface area of the triangle
            # area is the area of each triangle
            # area*mu is the area of each triangle projected in the direction of the observer
            # visibility is 0 for hidden, 0.5 for partial, 1.0 for visible
            # area*mu*visibility is the visibile projected area of each triangle (ie half the area for a partially-visible triangle)
            # so, intens_proj*area*mu*visibility is the intensity in the direction of the observer per the observed projected area of that triangle
            # and the sum of these values is the observed flux

            # note that the intensities are already projected but are per unit area
            # so we need to multiply by the /projected/ area of each triangle (thus the extra mu)
            return {'flux': np.sum(intens_proj_rel*area*mu*visibility)/((distance/10.)**2)+l3}


        elif method == 'IFM':
            # so far the function is kinda hollow
            vis2 = []
            vphase = []
            if len(dataset['ucoord_2']) == 0:
                vis2_2 = []
                vphase_2 = []
                vis2_3 = []
                vphase_3 = []
                t3_ampl = []
                t3_phase = []
            return dict(vis2=vis2, vphase=vphase, vis2_2=vis2_2,
                     vphase_2=vphase_2, vis2_3=vis2_3, vphase_3=vphase_3,
                     t3_ampl=t3_ampl, t3_phase=t3_phase)
        else:
            raise NotImplementedError("observe for dataset with method '{}' not implemented".format(method))


class Body(object):
    def __init__(self, comp_no, ind_self, ind_sibling, masses, ecc, datasets=[], dynamics_method='keplerian'):
        """
        TODO: add documentation
        """
        self._mesh_initialized = False
        self.dynamics_method = dynamics_method


        # TODO: eventually some of this stuff that assumes a BINARY orbit may need to be moved into
        # some subclass of Body (maybe BinaryBody).  These will want to be shared by Star and CustomBody,
        # but probably won't be shared by disk/ring-type objects

        # Let's remember the component number of this star in the parent orbit
        # 1 = primary
        # 2 = secondary
        self.comp_no = comp_no

        # We need to remember what index in all incoming position/velocity/euler
        # arrays correspond to both ourself and our sibling
        self.ind_self = ind_self
        self.ind_sibling = ind_sibling

        self.masses = masses
        self.ecc = ecc

        # compute q: notice that since we always do sibling_mass/self_mass, this
        # will automatically invert the value of q for the secondary component
        sibling_mass = self._get_mass_by_index(self.ind_sibling)
        self_mass = self._get_mass_by_index(self.ind_self)
        self.q = _value(sibling_mass / self_mass)

        # The basic float type is 'f8'
        ft = 'f8'

        # Pieter was being really general to allow 2-dimensional meshes in
        # the future, dim is probably always going to be 3
        dim = 3
        self.dim = dim

        # In the future we may send data in, but for now n_mesh will always be 0
        n_mesh = 0

        # self.mesh = None
        dtypes=[
                      ('center', ft, (dim, )),
                      ('size', ft),
                      ('triangle', ft, (3*dim, )),
                      ('normal_', ft, (dim, )),
                      ('velo___bol_', ft, (dim, )),
                      ('mu', ft),
                      ('visibility', ft),
                      ('comp_no', int)
                      ]

        self.mesh = np.zeros(n_mesh, dtype=dtypes)

        # TODO: double check to see if these are still used or can be removed
        self.time = None
        self.populated_at_time = []

        # Let's create a dictionary to store "standard" protomeshes at different "phases"
        # For example, we may want to store the mesh at periastron and use that as a standard
        # for reprojection for volume conservation in eccentric orbits.
        # Storing meshes should only be done through self.save_as_standard_mesh(theta)
        self._standard_meshes = {}

        # Let's create a dictionary to handle how each dataset should scale between
        # absolute and relative intensities.
        self._pblum_scale = {}

        # We'll also keep track of a conservative maximum r (from center of star to triangle, in real units).
        # This will be computed and stored when the periastron mesh is added as a standard
        self._max_r = None

        # TODO: allow custom meshes (see alpha:universe.Body.__init__)
        # TODO: reconsider partial/hidden/visible into opacity/visibility


    @property
    def mesh_initialized(self):
        """
        :return: whether the mesh has already been initialized
        :rtype: bool
        """
        return self._mesh_initialized

    @property
    def needs_recompute_instantaneous(self):
        """
        TODO: add documentation
        """
        # should be defined for any class that subclasses body if that body
        # can ever optimize by return false.

        # For example: stars can return False if they're in circular orbits
        return True

    @property
    def needs_volume_conservation(self):
        """
        TODO: add documentation
        """
        # should be defined for any class that subclasses body if that body
        # ever needs volume conservation (reprojection)

        # by default this will be False, but stars in non-circular orbits
        # need to return True

        # for any Body that does return True, a get_target_volume(self, etheta) must also be implemented
        return False

    @property
    def volume(self):
        """
        Compute volume of a mesh AT ITS CURRENT TIME/PROJECTION - this should be
        subclassed as needed for optimization or special cases

        :return: the current volume
        :rtype: float
        """

        return compute_volume(self.mesh['size'], self.mesh['center'], self.mesh['normal_'])

    @property
    def max_r(self):
        """
        Recall the maximum r (triangle furthest from the center of the star) of
        this star at periastron (when it is most deformed)

        :return: maximum r
        :rtype: float
        """
        return self._max_r

    @property
    def mass(self):
        return self._get_mass_by_index(self.ind_self)


    def _get_mass_by_index(self, index):
        """
        where index can either by an integer or a list of integers (returns some of masses)
        """
        if hasattr(index, '__iter__'):
            return sum([self.masses[i] for i in index])
        else:
            return self.masses[index]

    def _get_coords_by_index(self, coords_array, index):
        """
        where index can either by an integer or a list of integers (returns some of masses)
        coords_array should be a single array (xs, ys, or zs)
        """
        if hasattr(index, '__iter__'):
            # TODO: this should probably most definitely be mass-weighted (ie center of mass coordinates)
            return np.median([coords_array[i] for i in index])
        else:
            return coords_array[index]

    def instantaneous_distance(self, xs, ys, zs, sma):
        """
        TODO: add documentation
        """
        return np.sqrt(sum([(_value(self._get_coords_by_index(c, self.ind_self)) - _value(self._get_coords_by_index(c, self.ind_sibling)))**2 for c in (xs,ys,zs)])) / _value(sma)

    def initialize_mesh(self, **kwargs):
        """
        TODO: add documentation

        optional kwargs for BRS marching if time-dependent: (masses, sma)
        """
        # TODO: accept theta as an argument (compute d instead of assume d=1-e), for now will assume at periastron

        mesh_method = kwargs.get('mesh_method', self.mesh_method)

        # now let's do all the stuff that is potential-dependent
        d=1-self.ecc
        the_grid, scale, mesh_args = self._build_mesh(d=d, mesh_method=mesh_method, **kwargs)
        self._scale = scale
        self._mesh_args = mesh_args

        N = len(the_grid)

        if mesh_method == 'marching':
            maxpoints = int(kwargs.get('maxpoints', self.maxpoints))
            if N>=(maxpoints-1):
                raise ValueError(("Maximum number of triangles reached ({}). "
                                  "Consider raising the value of the parameter "
                                  "'maxpoints', or "
                                  "decrease the mesh density. It is also "
                                  "possible that the equipotential surface is "
                                  "not closed.").format(N))


        logger.info("covered surface with %d triangles"%(N))

        dtypes = self.mesh.dtype
        self.mesh = np.zeros(N,dtype=dtypes)
        self.mesh['center'] = the_grid[:,0:3]
        self.mesh['size'] = the_grid[:,3]
        self.mesh['triangle'] = the_grid[:,4:13]
        self.mesh['normal_'] = the_grid[:,13:16]
        self.mesh['comp_no'] = self.comp_no

        #self._fill_normals()  # TESTING

        self._fill_abun(kwargs.get('abun', self.abun))  # subclassed objects must set self.abun before calling initialize_mesh
        self._compute_instantaneous_quantities([], [], [], d=d) # TODO: is this Star-specific
        self._fill_logg([], [], [], d=d)
        self._fill_grav()
        self._fill_teff()

        self._mesh_initialized = True

        self.save_as_standard_mesh(theta=0.0)

        mesh = self.get_standard_mesh(scaled=True)  # TODO: set to require periastron, when available

        self.volume_at_periastron = compute_volume(mesh['size'], mesh['center'], mesh['normal_'])

        return

    def save_as_standard_mesh(self, theta=0.0):
        """
        TODO: add documentation
        """
        # TODO: standards don't need to store local quantities, just mesh stuffs
        # TODO: change from theta to d?

        self._standard_meshes[theta] = self.mesh.copy()

        if theta==0.0:
            # then this is when the object could be most inflated, so let's store
            # the maximum distance to a triangle.  This is then used to conservatively
            # and efficiently estimate whether an eclipse is possible at any given
            # combination of positions
            mesh = self.get_standard_mesh(theta=0.0, scaled=True)
            self._max_r = np.sqrt(max([x**2+y**2+z for x,y,z in mesh['center']]))

    def get_standard_mesh(self, theta=0.0, scaled=True):
        """
        TODO: add documentation
        """
        mesh = self._standard_meshes[theta].copy() if theta in self._standard_meshes.keys() else self.mesh.copy()

        if scaled:
            # TODO: would it be cheaper to just store a scaled and unscaled version rather than doing this each time?
            mesh['center'] *= self._scale
            mesh['size'] *= self._scale**2
            mesh['triangle'] *= self._scale

        return mesh

    def reset_position(self, time):
        """
        TODO: add documentation

        Reset the mesh to its original position.
        """
        # TODO: should this be renamed?  Or can it be removed?

        # we used to copy from _o_ to the current arrays, but now we hold
        # the standard meshes entirely independent and will later be pulling from them
        # so instead we'll reset visibility information in handle_eclipses

        self.time = time
        self.populated_at_time = []

    def update_position(self, time, xs, ys, zs, vxs, vys, vzs, ethetas, elongans, eincls, **kwargs):
        """
        Update the position of the star into its orbit

        :parameter float time: the current time
        :parameter list xs: a list/array of x-positions of ALL COMPONENTS in the :class:`System`
        :parameter list ys: a list/array of y-positions of ALL COMPONENTS in the :class:`System`
        :parameter list zs: a list/array of z-positions of ALL COMPONENTS in the :class:`System`
        :parameter list vxs: a list/array of x-velocities of ALL COMPONENTS in the :class:`System`
        :parameter list vys: a list/array of y-velocities of ALL COMPONENTS in the :class:`System`
        :parameter list vzs: a list/array of z-velocities of ALL COMPONENTS in the :class:`System`
        :parameter list ethetas: a list/array of euler-thetas of ALL COMPONENTS in the :class:`System`
        :parameter list elongans: a list/array of euler-longans of ALL COMPONENTS in the :class:`System`
        :parameter list eincls: a list/array of euler-incles of ALL COMPONENTS in the :class:`System`
        :raises NotImplementedError: if the dynamics_method is not supported
        """
        if not self.mesh_initialized:
            self.initialize_mesh()

        self.reset_position(time)
        mesh = self.get_standard_mesh(scaled=True)  # TODO: use nearest theta/d once supported
        mesh_unscaled = self.get_standard_mesh(scaled=False)

        #-- Volume Conservation
        if self.needs_volume_conservation:
            target_volume = self.get_target_volume(ethetas[self.ind_self])
            mesh_table = np.column_stack([mesh_unscaled[x] for x in ('center','size','triangle','normal_')])


            logger.info("volume conservation: target_volume={}".format(target_volume))

            # TODO: this seems Star-specific - should it be moved to that class or can it be generalized?

            potential, d, q, F, Phi = self._mesh_args

            # override d to be the current value
            d = self.instantaneous_distance(xs, ys, zs, self.sma)

            # Now we need to override Phi to be the Phi that gives target_volume
            # This can be done in one of several ways (eventually this might become
            # an option).

            ################  ROBUST NEWTON-RAPHSON APPROACH  #################
            def match_volume(Phi, mesh_table, potential, d, q, F, target_volume):
                mesh_args = (potential, d, q, F, Phi)
                new_table = cmarching.reproject(mesh_table, *mesh_args)

                centers = new_table[:,0:3]*self._scale
                #sizes = new_table[:,3]*self._scale**2

                normals = new_table[:,13:16]

                # TODO: not sure why sizes are all 0... but for now we can get them this way
                triangles = new_table[:,4:13]*self._scale
                sizes = fgeometry.compute_sizes(triangles)

                new_volume = compute_volume(sizes, centers, normals)
                #print "***", Phi, new_volume, target_volume, max(sizes)

                return new_volume - target_volume


            Phi = newton(match_volume, x0=Phi, args=(mesh_table, potential, d, q, F, target_volume), tol=1e-4, maxiter=1000)

            # to store this as instantaneous pot, we need to translate back to the secondary ref frame if necessary
            if self.comp_no == 2:
                # TODO: may need to invert this equation?
                self._instantaneous_pot = self.q*Phi - 0.5 * (self.q-1)
            else:
                self._instantaneous_pot = Phi
            ###################################################################


            ####### TODO: ANALYTIC/INTERPOLATION APPROXIMATION APPROACH #######

            ###################################################################


            #-- Reprojection
            logger.info("reprojecting mesh onto Phi={} at d={}".format(Phi, d))

            # here we send the unscaled standard mesh in to be reprojected onto the new potential
            mesh_args = (potential, d, q, F, Phi)
            mesh_table = cmarching.reproject(mesh_table, *mesh_args) #*self._scale

            # now we'll apply this reprojected mesh to the scaled mesh.  Since
            # mesh_table is unscaled, we'll rescale them here before setting them.
            mesh['center'] = mesh_table[:,0:3]*self._scale
            mesh['size'] = mesh_table[:,3]*self._scale**2
            mesh['triangle'] = mesh_table[:,4:13]*self._scale
            mesh['normal_'] = mesh_table[:,13:16]
            # TODO: do we need to update volo__bol_?
            # TODO: do we need to update mu?


        #-- Update position to be in orbit
        if self.dynamics_method == 'keplerian':
            # if we can't get the polar direction, assume it's in the negative Z-direction
            try:
                # TODO: implement get_polar_direction (see below for alpha version)
                # this will currently ALWAYS fail and follow the except - meaning
                # misaligned orbits are not yet supported
                polar_dir = -self.get_polar_direction(norm=True)
            except:
                #logger.warning("assuming polar direction - misaligned orbits not yet supported")
                polar_dir = default_polar_dir

            pos = (_value(xs[self.ind_self]), _value(ys[self.ind_self]), _value(zs[self.ind_self]))
            vel = (_value(vxs[self.ind_self]), _value(vys[self.ind_self]), _value(vzs[self.ind_self]))
            euler = (_value(ethetas[self.ind_self]), _value(elongans[self.ind_self]), _value(eincls[self.ind_self]))


            # Now let's transform from the star's coordinate system to the system's
            # coordinate system.  Note here that the second argument (center) is used
            # to overwrite those values, whereas the sevent argument (also center) is
            # used simply for input.  The fact that they're the same shouldn't cause
            # any problems.
            # This function will /alter/ the values in each of the input arrays,
            # and therefore does not return anything.
            #
            # TODO: rewrite this to return altered mesh instead of changing in memory (unless that causes performance issues)
            # TODO: make sure this handles systemic velocity/proper motion correctly (input pos and vel should be sent in with corrections already (once implemented))
            ctrans.place_in_binary_orbit(mesh['mu'], mesh['center'],
                                     mesh['triangle'], mesh['normal_'],
                                     mesh['velo___bol_'], polar_dir*self.freq_rot,
                                     mesh['center'], euler, pos, vel)

        else:
            # TODO: need nbody version of this (need to check dynamics_method)!!!  (probably won't handle rotation and might not have euler angles?)
            raise NotImplementedError("update_position for dynamics_method={} not supported".format(self.dynamics_method))


        # Finally, update the mesh and return
        self.mesh = mesh

        #-- Normals and centers are updated, but sizes are not.
        # TODO: why do we need to update sizes?  Let's check to see if this can be skipped
        # for some reason this does seem to be necessary for eccentric orbits at the moment (reprojection is setting sizes to 0)
        self._fill_sizes()

        # check if instantaneous quantities need to be (re)computed
        if 'logg' not in self.mesh.dtype.names or self.needs_recompute_instantaneous:
            self._compute_instantaneous_quantities(xs, ys, zs)

            # Now fill local instantaneous quantities
            self._fill_logg(xs, ys, zs)
            self._fill_grav()
            self._fill_teff()

        return

    def compute_luminosity(self, dataset, **kwargs):
        """
        """
        mesh = self.mesh

        # areas are the NON-projected areas of each surface element.  We'll be
        # integrated over normal intensities, so we don't need to worry about
        # multiplying by mu to get projected areas.
        areas = (mesh['size']*u.solRad**3).to(u.m**3).value

        # intens_norm_abs are directly out of the passbands module and are
        # emergent normal intensities in this dataset's passband/atm in absolute units
        intens_norm_abs = mesh['intens_norm_abs:{}'.format(dataset)]

        # The luminosity will be the integrated NORMAL intensities, but since some
        # limbdarkening laws still darken/brighten at mu=0, we'll compute the
        # limbdarkening for each element as if it were at mu=0 (directly along
        # the normal).
        #ld_coeffs = kwargs.get('ld_coeffs', [0.0,0.0])
        #ld_func = kwargs.get('ld_func', 'logarithmic')
        #ld = getattr(limbdark, 'ld_{}'.format(ld_func))(np.zeros(len(mesh['mu'])), ld_coeffs)
        ld = 1.0

        # Our total integrated intensity in absolute units (luminosity) is now
        # simply the sum of the normal emergent intensities times pi (to account
        # for intensities emitted in all directions across the solid angle),
        # limbdarkened as if they were at mu=0, and multiplied by their respective areas
        total_integrated_intensity = np.sum(intens_norm_abs*ld*areas) * np.pi

        return total_integrated_intensity * self.get_pblum_scale(dataset)

    def compute_pblum_scale(self, dataset, pblum, **kwargs):
        """
        intensities should already be computed for this dataset at the time for which pblum is being provided

        TODO: add documentation
        """

        total_integrated_intensity = self.compute_luminosity(dataset, **kwargs)


        # We now want to remember the scale for all intensities such that the
        # luminosity in relative units gives the provided pblum
        pblum_scale = pblum / total_integrated_intensity

        self._pblum_scale[dataset] = pblum_scale

    def get_pblum_scale(self, dataset):
        """
        """
        if dataset in self._pblum_scale.keys():
            return self._pblum_scale[dataset]
        else:
            #logger.warning("no pblum scale found for dataset: {}".format(dataset))
            return 1.0


    def populate_observable(self, time, method, dataset, **kwargs):
        """
        TODO: add documentation
        """

        if method in ['MESH']:
            return

        if time==self.time and dataset in self.populated_at_time:
            # then we've already computed the needed columns

            # TODO: handle the case of intensities already computed by /different/ dataset (ie RVs computed first and filling intensities and then LC requesting intensities with SAME passband/atm)
            return

        new_mesh_cols = getattr(self, '_populate_{}'.format(method.lower()))(dataset, **kwargs)

        for key, col in new_mesh_cols.items():
            #print "*** {}:{}".format(key, dataset)
            self._fill_column("{}:{}".format(key, dataset), col)

        self.populated_at_time.append(dataset)

    def _fill_column(self, columnname, data, inds=None):
        """
        TODO: add documentation
        """

        if isinstance(data, float):
            # then we have a single value - this will be fine for self.mesh[columname] = data,
            # but not any of the other cases
            data = np.ones(len(self.mesh['mu']))*data  # TODO: this will only work for floats

        #data = data.copy()

        if columnname in self.mesh.dtype.names:
            if inds is None:
                self.mesh[columnname] = data
            else:
                self.mesh[columnname][~inds] = np.nan
                self.mesh[columnname][inds] = data
        else:
            # now add this as a new field to the mesh with the field name given by dataset
            if inds is None:
                # TODO: could try to optimize this by "predicting" which cols will be necessary in the init and creating the mesh with them included

                # NOTE: mlab is >3 times faster than recfunctions in simple benchmarking cases, but I can't seem to get the data into the mesh correctly

                self.mesh = recfunctions.append_fields(self.mesh, columnname, data)
                # to avoid having to do this expensive operation at every step, let's
                # append empty columns to each of the standard meshes so the columns
                # will exist at the next timestep
                for k,standard_mesh in self._standard_meshes.items():
                    self._standard_meshes[k] = recfunctions.append_fields(standard_mesh, columnname, np.zeros(len(data)))


                #dtype = np.dtype([(columnname, 'f8')])
                #fill = np.zeros(len(self.mesh), dtype=dtype)
                #self.mesh = mlab.rec_append_fields(self.mesh, columnname, fill[columnname])
                #self.mesh[columnname] = data

            else:
                arr = np.zeros(len(self.mesh['mu']))
                arr[~inds] = np.nan
                arr[inds] = data

                # TODO: apply to standard meshes as above (probably move the actually appending to below this inner if-else)

                # NOTE: mlab is >3 times faster than recfunctions in simple benchmarking cases
                self.mesh = recfunctions.append_fields(self.mesh, columnname, arr)
                ## self.mesh = mlab.rec_append_fields(self.mesh, columnname, arr)


    def _fill_centers(self):
        """
        Compute the centers of the triangles and update the values in the CURRENT mesh.
        """
        # TODO: get rid of this method?
        for idim in range(self.dim):
            #self.mesh['_o_center'][:,idim] = self.mesh['_o_triangle'][:,idim::self.dim].sum(axis=1)/3.
            self.mesh['center'][:,idim] = self.mesh['triangle'][:,idim::self.dim].sum(axis=1)/3.

    def _fill_sizes(self):
        r"""
        Compute triangle sizes from the triangle vertices.

        The size :math:`s` of a triangle with vertex coordinates
        :math:`\vec{e}_0`, :math:`\vec{e}_1` and :math:`\vec{e}_2` is given by

        Let

        .. math::

            \vec{s}_0 = \vec{e}_0 - \vec{e}_1 \\
            \vec{s}_1 = \vec{e}_0 - \vec{e}_2 \\
            \vec{s}_2 = \vec{e}_1 - \vec{e}_2 \\

        then

        .. math::

            a = ||\vec{s}_0|| \\
            b = ||\vec{s}_1|| \\
            c = ||\vec{s}_2|| \\
            k = \frac{a + b + c}{2} \\

        and finally:

        .. math::

            s = \sqrt{ k (k-a) (k-b) (k-c)}

        """
        # TODO: get rid of this method (currently is used since marching doesn't seem to return the correct sizes)
        self.mesh['size'] = fgeometry.compute_sizes(self.mesh['triangle'])
        return

    def _fill_normals(self):
        """
        Compute normals from the triangle vertices.

        The normal on a triangle is defined as the cross product of the edge
        vectors:

        .. math::

            \vec{n} = \vec{e}_{0,1} \times \vec{e}_{0,2}

        with

        .. math::

            \vec{e}_{0,1} = \vec{e}_0 - \vec{e}_1 \\
            \vec{e}_{0,2} = \vec{e}_0 - \vec{e}_2 \\

        Comparison between numeric normals and true normals for a sphere. The
        numeric normals are automatically computed for an Ellipsoid. For a
        sphere, they are easily defined. The example below doesn't work anymore
        because the Ellipsoid class has been removed. Perhaps we re-include it
        again at some point.

        >>> #sphere = Ellipsoid()
        >>> #sphere.compute_mesh(delta=0.3)

        >>> #p = mlab.figure()
        >>> #sphere.plot3D(normals=True)

        >>> #sphere.mesh['normal_'] = sphere.mesh['center']
        >>> #sphere.plot3D(normals=True)
        >>> #p = mlab.gcf().scene.camera.parallel_scale=0.3
        >>> #p = mlab.draw()

        """
        # TODO: remove this method?
        # Normal is cross product of two sides


        # Compute the sides
        side1 = self.mesh['triangle'][:,0*self.dim:1*self.dim] -\
                self.mesh['triangle'][:,1*self.dim:2*self.dim]
        side2 = self.mesh['triangle'][:,0*self.dim:1*self.dim] -\
                self.mesh['triangle'][:,2*self.dim:3*self.dim]

        # Compute the cross product
        self.mesh['normal_'] = -np.cross(side1, side2)




class CustomBody(Body):
    def __init__(self, masses, sma, ecc, freq_rot, teff, abun, dynamics_method='keplerian', ind_self=0, ind_sibling=1, comp_no=1, datasets=[], **kwargs):
        """
        [NOT IMPLEMENTED]

        :parameter masses: mass of each component (solMass)
        :type masses: list of floats
        :parameter float sma: sma of this component's parent orbit (solRad)
        :parameter float freq_rot: rotation frequency (1/d)
        :parameter float abun: abundance of this star
        :parameter int ind_self: index in all arrays (positions, masses, etc) for this object
        :parameter int ind_sibling: index in all arrays (positions, masses, etc)
            for the sibling of this object
        :return: instantiated :class:`CustomBody` object
        :raises NotImplementedError: because it isn't
        """
        super(CustomBody, self).__init__(comp_no, ind_self, ind_sibling, masses, ecc, datasets, dynamics_method=dynamics_method)


        self.teff = teff
        self.abun = abun


        self.sma = sma


        raise NotImplementedError


    @classmethod
    def from_bundle(cls, b, component, compute=None, dynamics_method='keplerian', datasets=[], **kwargs):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        # TODO: handle overriding options from kwargs


        hier = b.hierarchy

        if not len(hier.get_value()):
            raise NotImplementedError("Star meshing requires a hierarchy to exist")


        label_self = component
        label_sibling = hier.get_stars_of_sibling_of(component)
        label_orbit = hier.get_parent_of(component)
        starrefs  = hier.get_stars()

        ind_self = starrefs.index(label_self)
        # for the sibling, we may need to handle a list of stars (ie in the case of a hierarchical triple)
        ind_sibling = starrefs.index(label_sibling) if isinstance(label_sibling, str) else [starrefs.index(l) for l in label_sibling]
        comp_no = ['primary', 'secondary'].index(hier.get_primary_or_secondary(component))+1

        self_ps = b.filter(component=component, context='component')
        freq_rot = self_ps.get_value('freq', unit=u.rad/u.d)

        teff = b.get_value('teff', component=component, context='component', unit=u.K)

        abun = b.get_value('abun', component=component, context='component')

        masses = [b.get_value('mass', component=star, context='component', unit=u.solMass) for star in starrefs]
        sma = b.get_value('sma', component=label_orbit, context='component', unit=u.solRad)
        ecc = b.get_value('ecc', component=label_orbit, context='component')

        return cls(masses, sma, ecc, freq_rot, teff, abun, dynamics_method, ind_self, ind_sibling, comp_no, datasets=datasets)


    @property
    def needs_recompute_instantaneous(self):
        """
        CustomBody has all values fixed by default, so this always returns False

        :return: False
        """
        return False

    @property
    def needs_volume_conservation(self):
        """
        CustomBody will never reproject to handle volume conservation

        :return: False
        """
        return False


    def _build_mesh(self, d, **kwargs):
        """
        [NOT IMPLEMENTED]

        this function takes mesh_method and kwargs that came from the generic Body.intialize_mesh and returns
        the grid... intialize mesh then takes care of filling columns and rescaling to the correct units, etc

        :raises NotImplementedError: because it isn't
        """

        # if we don't provide instantaneous masses or smas, then assume they are
        # not time dependent - in which case they were already stored in the init
        masses = kwargs.get('masses', self.masses)  #solMass
        sma = kwargs.get('sma', self.sma)  # Rsol (same units as coordinates)
        q = self.q  # NOTE: this is automatically flipped to be 1./q for secondary components

        raise NotImplementedError

        return the_grid, sma, mesh_args


    def _fill_teff(self, **kwargs):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """

        self._fill_column('teff', self.teff)



    def _fill_abun(self, abun=0.0):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        self._fill_column('abun', abun)



    def _populate_ifm(self, dataset, **kwargs):
        """
        [NOT IMPLEMENTED]

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`

        :raises NotImplementedError: because it isn't
        """

        raise NotImplementedError

    def _populate_rv(self, dataset, **kwargs):
        """
        [NOT IMPLEMENTED]


        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`

        :raises NotImplementedError: because it isn't
        """

        raise NotImplementedError


    def _populate_lc(self, dataset, **kwargs):
        """
        [NOT IMPLEMENTED]

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`

        :raises NotImplementedError: because it isn't
        """

        raise NotImplementedError

        return {'intens_norm_abs': intens_norm_abs, 'intens_norm_rel': intens_norm_rel,
            'intens_proj_abs': intens_proj_abs, 'intens_proj_rel': intens_proj_rel}



class Star(Body):
    def __init__(self, F, Phi, masses, sma, ecc, freq_rot, teff, gravb_bol, gravb_law, abun, mesh_method='marching', dynamics_method='keplerian', ind_self=0, ind_sibling=1, comp_no=1, datasets=[], do_rv_grav=False, **kwargs):
        """

        :parameter float F: syncpar
        :parameter float Phi: equipotential of this star at periastron
        :parameter masses: mass of each component in the system (solMass)
        :type masses: list of floats
        :parameter float sma: sma of this component's parent orbit (solRad)
        :parameter float freq_rot: rotation frequency (1/d)
        :parameter float abun: abundance of this star
        :parameter int ind_self: index in all arrays (positions, masses, etc) for this object
        :parameter int ind_sibling: index in all arrays (positions, masses, etc)
            for the sibling of this object
        :return: instantiated :class:`Star` object
        """
        super(Star, self).__init__(comp_no, ind_self, ind_sibling, masses, ecc, datasets, dynamics_method=dynamics_method)

        # Remember how to compute the mesh
        self.mesh_method = mesh_method
        self.delta = kwargs.get('delta', 0.1)                               # Marching
        self.maxpoints = kwargs.get('maxpoints', 1e5)                       # Marching
        self.distortion_method = kwargs.get('distortion_method', 'roche')   # Marching (WD assumes roche)
        self.gridsize = kwargs.get('gridsize', 90)                          # WD

        self.do_rv_grav = do_rv_grav

        # Remember things we need to know about this star - these will all be used
        # as defaults if they are not passed in future calls.  If for some reason
        # they are time dependent, then the instantaneous values need to be passed
        # for each call to update_position
        self.F = F
        self.freq_rot = freq_rot
        self.sma = sma


        # compute Phi (Omega/pot): here again if we're the secondary star we have
        # to translate Phi since all meshing methods assume a primary component
        self.Phi_user = Phi  # this is the value set by the user (not translated)
        self._instantaneous_pot = Phi  # this is again the value set by the user but will be updated for eccentric orbits at each time
        if self.comp_no == 2:
            self.Phi = self.q*Phi - 0.5 * (self.q-1)
        else:
            self.Phi = Phi

        self.teff = teff
        self.gravb_bol = gravb_bol
        self.gravb_law = gravb_law
        self.abun = abun


        # Volume "conservation"
        self.volume_factor = 1.0  # TODO: eventually make this a parameter (currently defined to be the ratio between volumes at apastron/periastron)

        self._pbs = {}


    @classmethod
    def from_bundle(cls, b, component, compute=None, dynamics_method='keplerian', datasets=[], **kwargs):
        """
        Build a star from the :class:`phoebe.frontend.bundle.Bundle` and its
        hierarchy.

        Usually it makes more sense to call :meth:`System.from_bundle` directly.

        :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
        :parameter str component: label of the component in the bundle
        :parameter str compute: name of the computeoptions in the bundle
        :parameter str dynamics_method: method to use for computing the position
            of this star in the orbit
        :parameter list datasets: list of names of datasets
        :parameter **kwargs: temporary overrides for computeoptions
        :return: an instantiated :class:`Star` object
        """
        # TODO: handle overriding options from kwargs
        # TODO: do we need dynamics method???

        hier = b.hierarchy

        if not len(hier.get_value()):
            raise NotImplementedError("Star meshing requires a hierarchy to exist")


        label_self = component
        label_sibling = hier.get_stars_of_sibling_of(component)
        label_orbit = hier.get_parent_of(component)
        starrefs  = hier.get_stars()

        ind_self = starrefs.index(label_self)
        # for the sibling, we may need to handle a list of stars (ie in the case of a hierarchical triple)
        ind_sibling = starrefs.index(label_sibling) if isinstance(label_sibling, str) else [starrefs.index(l) for l in label_sibling]
        comp_no = ['primary', 'secondary'].index(hier.get_primary_or_secondary(component))+1

        # meshing for BRS needs d,q,F,Phi
        # d is instantaneous based on x,y,z of self and sibling
        # q is instantaneous based on masses of self and sibling
        # F we can get now
        # Phi we can get now

        self_ps = b.filter(component=component, context='component')
        F = self_ps.get_value('syncpar')
        Phi = self_ps.get_value('pot')
        freq_rot = self_ps.get_value('freq', unit=u.rad/u.d)
        # NOTE: we need F for roche geometry (marching, reprojection), but freq_rot for ctrans.place_in_orbit


        masses = [b.get_value('mass', component=star, context='component', unit=u.solMass) for star in starrefs]
        sma = b.get_value('sma', component=label_orbit, context='component', unit=u.solRad)
        ecc = b.get_value('ecc', component=label_orbit, context='component')

        teff = b.get_value('teff', component=component, context='component', unit=u.K)
        gravb_law = b.get_value('gravblaw', component=component, context='component')
        gravb_bol= b.get_value('gravb_bol', component=component, context='component')

        abun = b.get_value('abun', component=component, context='component')

        try:
            do_rv_grav = b.get_value('rv_grav', component=component, compute=compute, check_relevant=False, **kwargs) if compute is not None else False
        except ValueError:
            # rv_grav may not have been copied to this component if no rvs are attached
            do_rv_grav = False

        # pass kwargs in case mesh_method was temporarily overriden
        mesh_method = b.get_value('mesh_method', component=component, compute=compute, **kwargs) if compute is not None else 'marching'

        mesh_kwargs = {}
        if mesh_method == 'marching':
            mesh_kwargs['delta'] = b.get_value('delta', component=component, compute=compute) if compute is not None else 0.1
            mesh_kwargs['maxpoints'] = b.get_value('maxpoints', component=component, compute=compute) if compute is not None else 1e5
            mesh_kwargs['distortion_method'] = b.get_value('distortion_method', component=component, compute=compute) if compute is not None else 'roche'
        elif mesh_method == 'wd':
            mesh_kwargs['gridsize'] = b.get_value('gridsize', component=component, compute=compute) if compute is not None else 30
        else:
            raise NotImplementedError

        return cls(F, Phi, masses, sma, ecc, freq_rot, teff, gravb_bol, gravb_law,
                abun, mesh_method, dynamics_method, ind_self, ind_sibling, comp_no,
                datasets=datasets, do_rv_grav=do_rv_grav, **mesh_kwargs)


    @property
    def needs_recompute_instantaneous(self):
        """
        TODO: add documentation
        """
        if self.ecc != 0.0:
            # for eccentric orbits we need to recompute values at every time-step
            return True
        else:
            # In circular orbits we should be safe to assume these quantities
            # remain constant

            # TODO: may need to add conditions here for reflection/heating or
            # if time-dependent parameters are passed
            return False

    @property
    def needs_volume_conservation(self):
        """
        TODO: add documentation

        we can skip volume conservation only for circular orbits
        """
        return self.ecc != 0


    def get_target_volume(self, etheta):
        """
        TODO: add documentation

        get the volume that the BRS should have at a given euler theta
        """
        # TODO: make this a function of d instead of etheta?
        logger.info("determining target volume at theta={}".format(etheta))

        # TODO: eventually this could allow us to "break" volume conservation and have volume be a function of d,
        # with some scaling factor provided by the user as a parameter.  Until then, we'll assume volume is conserved
        # which means the volume should always be the same as it was defined at periaston.

        return self.volume_at_periastron

    def _build_mesh(self, d, mesh_method, **kwargs):
        """
        this function takes mesh_method and kwargs that came from the generic Body.intialize_mesh and returns
        the grid... intialize mesh then takes care of filling columns and rescaling to the correct units, etc
        """

        # if we don't provide instantaneous masses or smas, then assume they are
        # not time dependent - in which case they were already stored in the init
        masses = kwargs.get('masses', self.masses)  #solMass
        sma = kwargs.get('sma', self.sma)  # Rsol (same units as coordinates)
        F = kwargs.get('F', self.F)
        Phi = kwargs.get('Phi', self.Phi)  # NOTE: self.Phi automatically corrects for the secondary star
        q = self.q  # NOTE: this is automatically flipped to be 1./q for secondary components

        if mesh_method == 'marching':
            delta = kwargs.get('delta', self.delta)
            # NOTE: delta needs to be rescaled (see below in roche distortion method)
            maxpoints = int(kwargs.get('maxpoints', self.maxpoints))

            if self.distortion_method == 'roche':
                # TODO: check whether roche or misaligned roche from values of incl, etc!!!!

                potential = 'BinaryRoche'
                mesh_args = (potential, d, q, F, Phi)

                # TODO: should we take r_pole from the PS (as a passed argument) or
                # compute it here? - this is probably the instantaneous r_pole?
                # TODO: all this seems to do is set delta... is that all it should do?
                r_pole = project_onto_potential(np.array((0,0,1e-5)), *mesh_args).r
                r_pole_= np.linalg.norm(r_pole)
                delta = delta*r_pole_

                the_grid = cmarching.discretize(delta, maxpoints, *mesh_args)[:-2]

            elif self.distortion_method == 'sphere':
                # TODO: implement this (discretize and save mesh_args)
                raise NotImplementedError("sphere distortion method not yet supported - try roche")
            elif self.distortion_method == 'nbody':
                # TODO: implement this (discretize and save mesh_args)
                raise NotImplementedError("nbody distortion method not yet supported - try roche")
            else:
                raise NotImplementedError

        elif mesh_method == 'wd':
            # there is no distortion_method for WD - it must be roche
            potential = 'BinaryRoche'
            mesh_args = (potential, d, q, F, Phi)

            N = int(kwargs.get('gridsize', self.gridsize))

            the_grid = discretize_wd_style(N, *mesh_args)


        else:
            raise NotImplementedError("mesh method '{}' is not supported".format(mesh_method))

        # return the_grid, scale, mesh_args
        return the_grid, sma, mesh_args

    def _compute_instantaneous_quantities(self, xs, ys, zs, **kwargs):
        """
        TODO: add documentation
        """

        # TODO: check whether we want the automatically inverted q or not
        q = self.q  # NOTE: this is automatically flipped to be 1./q for secondary components
        d = kwargs.get('d') if 'd' in kwargs.keys() else self.instantaneous_distance(xs, ys, zs, self.sma)

        r_pole_ = project_onto_potential(np.array((0, 0, 1e-5)), *self._mesh_args).r # instantaneous unitless r_pole (not rpole defined at periastron)
        g_pole = np.sqrt(dBinaryRochedx(r_pole_, d, q, self.F)**2 + dBinaryRochedz(r_pole_, d, q, self.F)**2)
        rel_to_abs = constants.G.si.value*constants.M_sun.si.value*self.masses[self.ind_self]/(self.sma*constants.R_sun.si.value)**2*100. # 100 for m/s**2 -> cm/s**2

        self._instantaneous_gpole = g_pole * rel_to_abs
        self._instantaneous_rpole = np.sqrt((r_pole_*r_pole_).sum())


    def _fill_logg(self, xs, ys, zs, **kwargs):
        """
        TODO: add documentation

        Calculate local surface gravity

        GMSunNom = 1.3271244e20 m**3 s**-2
        RSunNom = 6.597e8 m
        """

        # TODO: check whether we want the automatically inverted q or not
        q = self.q  # NOTE: this is automatically flipped to be 1./q for secondary components
        d = kwargs.get('d') if 'd' in kwargs.keys() else self.instantaneous_distance(xs, ys, zs, self.sma)

        # TODO: incorporate these into astropy.constants.
        #~ GMSunNom = 1.3271244e20
        #~ RSunNom = 6.597e8
        #~ rel_to_abs = GMSunNom*self.masses[self.ind_self]/(self.sma*RSunNom)**2*100. # 100 for m/s**2 -> cm/s**2
        rel_to_abs = constants.G.si.value*constants.M_sun.si.value*self.masses[self.ind_self]/(self.sma*constants.R_sun.si.value)**2*100. # 100 for m/s**2 -> cm/s**2


        # Compute gradients:
        mesh = self.get_standard_mesh(scaled=False)
        dOmegadx, dOmegady, dOmegadz = roche.binary_potential_gradient(mesh['center'][:,0], mesh['center'][:,1], mesh['center'][:,2], q, d, self.F, normalize=False, output_type='list')

        logg = np.log10(rel_to_abs * np.sqrt(dOmegadx**2+dOmegady**2+dOmegadz**2))
        self._fill_column('logg', logg)

        logger.info("derived surface gravity: %.3f <= log g<= %.3f (g_p=%s and Rp=%s Rsol)"%(self.mesh['logg'].min(),self.mesh['logg'].max(),self._instantaneous_gpole,self._instantaneous_rpole*self._scale))

    def _fill_grav(self, **kwargs):
        """
        TODO: add documentation

        requires _fill_logg to have been called
        """

        grav = abs(10**(self.mesh['logg']-2)/self._instantaneous_gpole)**self.gravb_bol

        self._fill_column('grav', grav)

    def _fill_teff(self, **kwargs):
        r"""

        requires _fill_logg and _fill_grav to have been called

        Calculate local temperatureof a BinaryRocheStar.

        If the law of [Espinosa2012]_ is used, some approximations are made:

            - Since the law itself is too complicated too solve during the
              computations, the table with approximate von Zeipel exponents from
              [Espinosa2012]_ is used.
            - The two parameters in the table are mass ratio :math:`q` and
              filling factor :math:`\rho`. The latter is defined as the ratio
              between the radius at the tip, and the first Lagrangian point.
              As the Langrangian points can be badly defined for weird
              configurations, we approximate the Lagrangian point as 3/2 of the
              polar radius (inspired by break up radius in fast rotating stars).
              This is subject to improvement!

        """

        if self.gravb_law == 'espinosa':
            # TODO: check whether we want the automatically inverted q or not
            q = self.q  # NOTE: this is automatically flipped to be 1./q for secondary components
            F = self.syncpar
            sma = self.sma

            # To compute the filling factor, we're gonna cheat a little bit: we
            # should compute the ratio of the tip radius to the first Lagrangian
            # point. However, L1 might be poorly defined for weird geometries
            # so we approximate it as 1.5 times the polar radius.
            rp = self._instantaneous_rpole  # should be in Rsol
            maxr = coordinates.norm(self.get_standard_mesh(scaled=True)['center'],axis=1).max()

            L1 = roche.exact_lagrangian_points(q, F=F, d=1.0, sma=sma)[0]
            rho = maxr / L1
            gravb = roche.zeipel_gravb_binary()(np.log10(q), rho)[0][0]
            # TODO: why did Pieter set this back to the parameter?
        #     self.params['component']['gravb'] = gravb
            logger.info("gravb(Espinosa): F = {}, q = {}, filling factor = {} --> gravb = {}".format(F, q, rho, gravb))
            if gravb>1.0 or gravb<0:
                raise ValueError('Invalid gravity darkening parameter beta={}'.format(gravb))

        elif self.gravb_law == 'claret':
            teff = np.log10(self.teff)
            logg = np.log10(self._instantaneous_gpole*100)
            abun = self.abun
            axv, pix = roche.claret_gravb()
            gravb = interp_nDgrid.interpolate([[teff], [logg], [abun]], axv, pix)[0][0]
            logger.info('gravb(Claret): teff = {:.3f}, logg = {:.6f}, abun = {:.3f} ---> gravb = {:.3f}'.format(10**teff, logg, abun, gravb))
            # TODO: why did Pieter set this back to the parameter?
        #     self.params['component']['gravb'] = gravb

        # Now use the Zeipel law:

        if 'teffpolar' in kwargs.keys():
            Teff = kwargs['teffpolar']
            typ = 'polar'
        else:
            Teff = kwargs.get('teff', self.teff)
            typ = 'mean'

        # Consistency check for gravity brightening
        if Teff >= 8000. and self.gravb_bol < 0.9:
            logger.info('Object probably has a radiative atm (Teff={:.0f}K>8000K), for which gravb=1.00 might be a better approx than gravb={:.2f}'.format(Teff,self.gravb_bol))
        elif Teff <= 6600. and self.gravb_bol >= 0.9:
            logger.info('Object probably has a convective atm (Teff={:.0f}K<6600K), for which gravb=0.32 might be a better approx than gravb={:.2f}'.format(Teff,self.gravb_bol))
        elif self.gravb_bol < 0.32 or self.gravb_bol > 1.00:
            logger.info('Object has intermittent temperature, gravb should be between 0.32-1.00')

        # Compute G and Tpole
        Grav = self.mesh['grav']


        if typ == 'mean':
            Tpole = Teff*(np.sum(self.mesh['size']) / np.sum(Grav*self.mesh['size']))**(0.25)
        elif typ == 'polar':
            Tpole = Teff
        else:
            raise ValueError("Cannot interpret temperature type '{}' (needs to be one of ['mean','polar'])".format(typ))

        self._instantaneous_teffpole = Tpole




        # Now we can compute the local temperatures.
        teff = (Grav * Tpole**4)**0.25
        self._fill_column('teff', teff)

        #=== FORTRAN ATTEMPT DOESN'T GO ANY FASTER =======
        #Tpole,system.mesh['teff'] = froche.temperature_zeipel(system.mesh['logg'],
        #                   system.mesh['size'], Teff, ['mean','polar'].index(type),
        #                   beta, gp)
        logger.info("derived effective temperature (Zeipel) (%.3f <= teff <= %.3f, Tp=%.3f)"%(self.mesh['teff'].min(),self.mesh['teff'].max(),Tpole))

    def _fill_abun(self, abun=0.0):
        """
        TODO: add documentation
        """
        self._fill_column('abun', abun)


    def _populate_ifm(self, dataset, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`
        """

        # TODO
        # this is not correct - we need to compute ld for a
        # given effective wavelength, i.e
        # eff_wave = kwargs.get('eff_wave', 6562e-7)
        # passband = kwargs.get('passband', eff_wave)
        passband = kwargs.get('passband', 'Johnson:V')
        ld_coeffs = kwargs.get('ld_coeffs', [0.5,0.5])
        ld_func = kwargs.get('ld_func', 'logarithmic')
        atm = kwargs.get('atm', 'kurucz')
        boosting_alg = kwargs.get('boosting_alg', 'none')

        lc_cols = self._populate_lc(dataset, **kwargs)

        # LC cols should do for interferometry - at least for continuum
        cols = lc_cols

        return cols

    def _populate_rv(self, dataset, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`
        """

        # print "*** Star._populate_rv"
        passband = kwargs.get('passband', 'Johnson:V')
        ld_coeffs = kwargs.get('ld_coeffs', [0.5,0.5])
        ld_func = kwargs.get('ld_func', 'logarithmic')
        atm = kwargs.get('atm', 'kurucz')
        boosting_alg = kwargs.get('boosting_alg', 'none')

        # We need to fill all the flux-related columns so that we can weigh each
        # triangle's RV by its flux in the requested passband.
        lc_cols = self._populate_lc(dataset, **kwargs)

        # RV per element is just the z-component of the velocity vectory.  Note
        # the change in sign from our right-handed system to RV conventions.
        # These will be weighted by the fluxes when integrating
        rvs = -1*self.mesh['velo___bol_'][:,2]

        # Gravitational redshift
        if self.do_rv_grav:
            rv_grav = constants.G*(self.mass*u.solMass)/(self._instantaneous_rpole*u.solRad)/constants.c
            # rvs are in solrad/d internally
            rv_grav = rv_grav.to('solRad/d').value

            rvs += rv_grav

        cols = lc_cols
        cols['rv'] = rvs
        return cols


    def _populate_lc(self, dataset, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`

        :raises NotImplementedError: if lc_method is not supported
        """

        lc_method = kwargs.get('lc_method', 'numerical')  # TODO: make sure this is actually passed
        passband = kwargs.get('passband', 'Johnson:V')

        ld_coeffs = kwargs.get('ld_coeffs', [0.5,0.5])
        ld_func = kwargs.get('ld_func', 'logarithmic')
        atm = kwargs.get('atm', 'blackbody')
        boosting_alg = kwargs.get('boosting_alg', 'none')

        pblum = kwargs.get('pblum', 4*np.pi)

        mesh = self.mesh
        #mus = mesh['mu']

        if lc_method=='numerical':

            if passband not in self._pbs.keys():
                passband_fname = passbands._pbtable[passband]['fname']
                logger.info("using ptf file: {}".format(passband_fname))
                pb = passbands.Passband.load(passband_fname)

                self._pbs[passband] = pb

            # intens_norm_abs are the normal emergent passband intensities:
            intens_norm_abs = self._pbs[passband].Inorm(Teff=mesh['teff'], logg=mesh['logg'], met=mesh['abun'], atm=atm)

            # TODO: we only need this at t0 for compute_pblum_scale
            #self._instantaneous_Inormpole = self._pbs[passband].Inorm(self._instantaneous_teff, np.log10(self._instantaneous_gpole), atm=atm)

            # Handle pblum - distance and l3 scaling happens when integrating (in observe)
            # we need to scale each triangle so that the summed intens_norm_rel over the
            # entire star is equivalent to pblum / 4pi

            intens_norm_rel = intens_norm_abs * self.get_pblum_scale(dataset)


            # Beaming/boosting
            if boosting_alg == 'simple':
                raise NotImplementedError("'simple' boosting_alg not yet supported")
                # TODO: need to get alpha_b from the passband/atmosphere tables
                alpha_b = interp_boosting(atm_file, passband, atm_kwargs=atm_kwargs,
                                              red_kwargs=red_kwargs, vgamma=vgamma,
                                              interp_all=False)


            elif boosting_alg == 'local':
                raise NotImplementedError("'local' boosting_alg not yet supported")
                # TODO: need to get alpha_b from the passband/atmosphere tables
                alpha_b = interp_boosting(atm_file, passband, atm_kwargs=atm_kwargs,
                                              red_kwargs=red_kwargs, vgamma=vgamma)


            elif boosting_alg == 'global':
                raise NotImplementedError("'global' boosting_alg not yet supported")
                # TODO: need to get alpha_b from the passband/atmosphere tables
                alpha_b = interp_boosting(atm_file, passband, atm_kwargs=atm_kwargs,
                                              red_kwargs=red_kwargs, vgamma=vgamma)

            else:
                alpha_b = 0.0

            # light speed in Rsol/d
            # TODO: should we mutliply velo__bol_ by -1?
            ampl_boost = 1.0 + alpha_b * mesh['velo___bol_'][:,2]/37241.94167601236


            # Limb-darkening
            ld = getattr(limbdark, 'ld_{}'.format(ld_func))(mesh['mu'], ld_coeffs)

            # Apply boosting/beaming and limb-darkening to the projected intensities
            intens_proj_abs = intens_norm_abs * ld * ampl_boost
            intens_proj_rel = intens_norm_rel * ld * ampl_boost


            # TODO: handle reflection!!!! (see alpha:universe.py:generic_projected_intensity)
            #logger.warning("reflection/heating for fluxes not yet ported to beta")


        elif lc_method=='analytical':
            raise NotImplementedError("analytical fluxes not yet ported to beta")
            #lcdep, ref = system.get_parset(ref)
            # The projected intensity is normalised with the distance in cm, we need
            # to reconvert that into solar radii.
            #intens_proj = limbdark.sphere_intensity(body,lcdep)[1]/(constants.Rsol)**2

            # TODO: this probably needs to be moved into observe or backends.phoebe
            # (assuming it doesn't result in per-triangle quantities)

        else:
            raise NotImplementedError("lc_method '{}' not recognized".format(lc_method))


        # Take reddening into account (if given), but only for non-bolometric
        # passbands and nonzero extinction

        # TODO: reddening
        #logger.warning("reddening for fluxes not yet ported to beta")
        # if dataset != '__bol':

        #     # if there is a global reddening law
        #     red_parset = system.get_globals('reddening')
        #     if (red_parset is not None) and (red_parset['extinction'] > 0):
        #         ebv = red_parset['extinction'] / red_parset['Rv']
        #         proj_intens = reddening.redden(proj_intens,
        #                      passbands=[idep['passband']], ebv=ebv, rtype='flux',
        #                      law=red_parset['law'])[0]
        #         logger.info("Projected intensity is reddened with E(B-V)={} following {}".format(ebv, red_parset['law']))

        #     # if there is passband reddening
        #     if 'extinction' in idep and (idep['extinction'] > 0):
        #         extinction = idep['extinction']
        #         proj_intens = proj_intens / 10**(extinction/2.5)
        #         logger.info("Projected intensity is reddened with extinction={} (passband reddening)".format(extinction))



        # TODO: do we really need to store all of these if store_mesh==False?
        # Can we optimize by only returning the essentials if we know we don't need them?
        return {'intens_norm_abs': intens_norm_abs, 'intens_norm_rel': intens_norm_rel,
            'intens_proj_abs': intens_proj_abs, 'intens_proj_rel': intens_proj_rel,
            'ampl_boost': ampl_boost, 'ld': ld}


class Envelope(Body):
    def __init__(self, Phi, masses, sma, ecc, freq_rot, abun, mesh_method='marching',
            dynamics_method='keplerian', ind_self=0, ind_sibling=1, comp_no=1,
            datasets=[], do_rv_grav=False, **kwargs):
        """
        [NOT IMPLEMENTED]

        :parameter float Phi: equipotential of this star at periastron
        :parameter masses: mass of each component in the system (solMass)
        :type masses: list of floats
        :parameter float sma: sma of this component's parent orbit (solRad)
        :parameter float abun: abundance of this star
        :parameter int ind_self: index in all arrays (positions, masses, etc) for the primary star in this overcontact envelope
        :parameter int ind_sibling: index in all arrays (positions, masses, etc)
            for the secondary star in this overcontact envelope
        :return: instantiated :class:`Envelope` object
        """
        super(Envelope, self).__init__(comp_no, ind_self, ind_sibling, masses, ecc, datasets, dynamics_method=dynamics_method)

        # Remember how to compute the mesh
        self.mesh_method = mesh_method
        self.delta = kwargs.get('delta', 0.1)                               # Marching
        self.maxpoints = kwargs.get('maxpoints', 1e5)                       # Marching
        self.distortion_method = kwargs.get('distortion_method', 'roche')   # Marching (WD assumes roche)
        self.gridsize = kwargs.get('gridsize', 90)                          # WD

        self.do_rv_grav = do_rv_grav

        # Remember things we need to know about this star - these will all be used
        # as defaults if they are not passed in future calls.  If for some reason
        # they are time dependent, then the instantaneous values need to be passed
        # for each call to update_position
        self.F = 1.0 # by definition for an overcontact
        self.freq_rot = freq_rot   # TODO: change to just pass period and compute freq_rot here?
        self.sma = sma


        # compute Phi (Omega/pot): here again if we're the secondary star we have
        # to translate Phi since all meshing methods assume a primary component
        self.Phi_user = Phi  # this is the value set by the user (not translated)
        self._instantaneous_pot = Phi
        # for overcontacts, we'll always build the mesh from the primary star
        self.Phi = Phi

        #self.teff = teff
        #self.gravb_bol = gravb_bol
        #self.gravb_law = gravb_law
        self.abun = abun


        # Volume "conservation"
        self.volume_factor = 1.0  # TODO: eventually make this a parameter (currently defined to be the ratio between volumes at apastron/periastron)

        self._pbs = {}


    @classmethod
    def from_bundle(cls, b, component, compute=None, dynamics_method='keplerian', datasets=[], **kwargs):
        """
        [NOT IMPLEMENTED]

        Build an overcontact from the :class:`phoebe.frontend.bundle.Bundle` and its
        hierarchy.

        Usually it makes more sense to call :meth:`System.from_bundle` directly.

        :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
        :parameter str component: label of the component in the bundle
        :parameter str compute: name of the computeoptions in the bundle
        :parameter str dynamics_method: method to use for computing the position
            of this star in the orbit
        :parameter list datasets: list of names of datasets
        :parameter **kwargs: temporary overrides for computeoptions
        :return: an instantiated :class:`Envelope` object
        """
        # TODO: handle overriding options from kwargs
        # TODO: do we need dynamics method???

        hier = b.hierarchy

        if not len(hier.get_value()):
            raise NotImplementedError("Overcontact envelope meshing requires a hierarchy to exist")


        label_envelope = component
        # self is just the primary star in the same orbit
        label_self = hier.get_sibling_of(component)  # TODO: make sure this defaults to primary
        label_sibling = hier.get_sibling_of(label_self)  # TODO: make sure this defaults to secondary
        label_orbit = hier.get_parent_of(component)
        starrefs  = hier.get_stars()

        ind_self = starrefs.index(label_self)
        # for the sibling, we may need to handle a list of stars (ie in the case of a hierarchical triple)
        ind_sibling = starrefs.index(label_sibling) if isinstance(label_sibling, str) else [starrefs.index(l) for l in label_sibling]
        comp_no = 1

        # meshing for BRS needs d,q,F,Phi
        # d is instantaneous based on x,y,z of self and sibling
        # q is instantaneous based on masses of self and sibling
        # F we will assume is always 1 for an overcontact
        # Phi we can get now

        env_ps = b.filter(component=component, context='component')
        F = 1.0  # this is also hardcoded in the init, so isn't passed
        Phi = env_ps.get_value('pot')
        period = b.get_quantity(qualifier='period', unit=u.d, component=label_orbit, context='component')
        freq_rot = 2*np.pi*u.rad/period
        # NOTE: we need F for roche geometry (marching, reprojection), but freq_rot for ctrans.place_in_orbit


        masses = [b.get_value('mass', component=star, context='component', unit=u.solMass) for star in starrefs]
        sma = b.get_value('sma', component=label_orbit, context='component', unit=u.solRad)
        ecc = b.get_value('ecc', component=label_orbit, context='component')

        #teff = b.get_value('teff', component=component, context='component', unit=u.K)
        #gravb_law = b.get_value('gravblaw', component=component, context='component')
        #gravb_bol= b.get_value('gravb_bol', component=component, context='component')

        abun = b.get_value('abun', component=component, context='component')



        try:
            # TODO: will the rv_grav parameter ever be copied for the envelope?
            do_rv_grav = b.get_value('rv_grav', component=component, compute=compute, check_relevant=False, **kwargs) if compute is not None else False
        except ValueError:
            # rv_grav may not have been copied to this component if no rvs are attached
            do_rv_grav = False

        # pass kwargs in case mesh_method was temporarily overridden
        # TODO: make sure mesh_method copies for envelopes
        mesh_method = b.get_value('mesh_method', component=component, compute=compute, **kwargs) if compute is not None else 'marching'

        mesh_kwargs = {}
        if mesh_method == 'marching':
            mesh_kwargs['delta'] = b.get_value('delta', component=component, compute=compute) if compute is not None else 0.1
            mesh_kwargs['maxpoints'] = b.get_value('maxpoints', component=component, compute=compute) if compute is not None else 1e5
            mesh_kwargs['distortion_method'] = b.get_value('distortion_method', component=component, compute=compute) if compute is not None else 'roche'
        elif mesh_method == 'wd':
            mesh_kwargs['gridsize'] = b.get_value('gridsize', component=component, compute=compute) if compute is not None else 30
        else:
            raise NotImplementedError

        return cls(Phi, masses, sma, ecc, freq_rot, abun,
                mesh_method, dynamics_method, ind_self, ind_sibling, comp_no,
                datasets=datasets, do_rv_grav=do_rv_grav, **mesh_kwargs)

    @property
    def needs_recompute_instantaneous(self):
        """
        TODO: add documentation
        """
        if self.ecc != 0.0:
            # for eccentric orbits we need to recompute values at every time-step
            return True
        else:
            # In circular orbits we should be safe to assume these quantities
            # remain constant

            # TODO: may need to add conditions here for reflection/heating or
            # if time-dependent parameters are passed
            return False

    @property
    def needs_volume_conservation(self):
        """
        TODO: add documentation

        we can skip volume conservation only for circular orbits
        """
        return self.ecc != 0


    def get_target_volume(self, etheta):
        """
        TODO: add documentation

        get the volume that the BRS should have at a given euler theta
        """
        # TODO: make this a function of d instead of etheta?
        logger.info("determining target volume at theta={}".format(etheta))

        # TODO: eventually this could allow us to "break" volume conservation and have volume be a function of d,
        # with some scaling factor provided by the user as a parameter.  Until then, we'll assume volume is conserved
        # which means the volume should always be the same as it was defined at periaston.

        return self.volume_at_periastron

    def _build_mesh(self, d, mesh_method, **kwargs):
        """
        this function takes mesh_method and kwargs that came from the generic Body.intialize_mesh and returns
        the grid... intialize mesh then takes care of filling columns and rescaling to the correct units, etc
        """

        # if we don't provide instantaneous masses or smas, then assume they are
        # not time dependent - in which case they were already stored in the init
        masses = kwargs.get('masses', self.masses)  #solMass
        sma = kwargs.get('sma', self.sma)  # Rsol (same units as coordinates)
        F = kwargs.get('F', self.F)
        Phi = kwargs.get('Phi', self.Phi)
        q = self.q

        if mesh_method == 'marching':
            delta = kwargs.get('delta', self.delta)
            # NOTE: delta needs to be rescaled (see below in roche distortion method)
            maxpoints = int(kwargs.get('maxpoints', self.maxpoints))

            if self.distortion_method == 'roche':
                # TODO: check whether roche or misaligned roche from values of incl, etc!!!!

                potential = 'BinaryRoche'
                mesh_args = (potential, d, q, F, Phi)

                # TODO: should we take r_pole from the PS (as a passed argument) or
                # compute it here? - this is probably the instantaneous r_pole?
                # TODO: all this seems to do is set delta... is that all it should do?
                r_pole = project_onto_potential(np.array((0,0,1e-5)), *mesh_args).r
                r_pole_= np.linalg.norm(r_pole)
                delta = delta*r_pole_

                the_grid = cmarching.discretize(delta, maxpoints, *mesh_args)[:-2]

            elif self.distortion_method == 'sphere':
                # TODO: implement this (discretize and save mesh_args)
                raise NotImplementedError("sphere distortion method not yet supported - try roche")
            elif self.distortion_method == 'nbody':
                # TODO: implement this (discretize and save mesh_args)
                raise NotImplementedError("nbody distortion method not yet supported - try roche")
            else:
                raise NotImplementedError

        elif mesh_method == 'wd':
            # there is no distortion_method for WD - it must be roche
            potential = 'BinaryRoche'
            mesh_args = (potential, d, q, F, Phi)

            N = int(kwargs.get('gridsize', self.gridsize))

            the_grid = discretize_wd_style(N, *mesh_args)


        else:
            raise NotImplementedError("mesh method '{}' is not supported".format(mesh_method))

        # return the_grid, scale, mesh_args
        return the_grid, sma, mesh_args

    def _compute_instantaneous_quantities(self, xs, ys, zs, **kwargs):
        """
        TODO: add documentation
        """
        pass  # TODO: do we need any of these things for overcontacts?

        #q = self.q
        #d = kwargs.get('d') if 'd' in kwargs.keys() else self.instantaneous_distance(xs, ys, zs, self.sma)

        #r_pole_ = project_onto_potential(np.array((0, 0, 1e-5)), *self._mesh_args).r # instantaneous unitless r_pole (not rpole defined at periastron)
        #g_pole = np.sqrt(dBinaryRochedx(r_pole_, d, q, self.F)**2 + dBinaryRochedz(r_pole_, d, q, self.F)**2)
        #rel_to_abs = constants.G.si.value*constants.M_sun.si.value*self.masses[self.ind_self]/(self.sma*constants.R_sun.si.value)**2*100. # 100 for m/s**2 -> cm/s**2

        #self._instantaneous_gpole = g_pole * rel_to_abs
        #self._instantaneous_rpole = np.sqrt((r_pole_*r_pole_).sum())


    def _fill_logg(self, xs, ys, zs, **kwargs):
        """
        TODO: add documentation

        Calculate local surface gravity

        GMSunNom = 1.3271244e20 m**3 s**-2
        RSunNom = 6.597e8 m
        """
        self._fill_column('logg', 0.0)

        #q = self.q
        #d = kwargs.get('d') if 'd' in kwargs.keys() else self.instantaneous_distance(xs, ys, zs, self.sma)

        # TODO: incorporate these into astropy.constants.
        #~ GMSunNom = 1.3271244e20
        #~ RSunNom = 6.597e8
        #~ rel_to_abs = GMSunNom*self.masses[self.ind_self]/(self.sma*RSunNom)**2*100. # 100 for m/s**2 -> cm/s**2
        #rel_to_abs = constants.G.si.value*constants.M_sun.si.value*self.masses[self.ind_self]/(self.sma*constants.R_sun.si.value)**2*100. # 100 for m/s**2 -> cm/s**2


        # Compute gradients:
        #mesh = self.get_standard_mesh(scaled=False)
        #dOmegadx, dOmegady, dOmegadz = roche.binary_potential_gradient(mesh['center'][:,0], mesh['center'][:,1], mesh['center'][:,2], q, d, self.F, normalize=False, output_type='list')

        #logg = np.log10(rel_to_abs * np.sqrt(dOmegadx**2+dOmegady**2+dOmegadz**2))
        #self._fill_column('logg', logg)

        #logger.info("derived surface gravity: %.3f <= log g<= %.3f (g_p=%s and Rp=%s Rsol)"%(self.mesh['logg'].min(),self.mesh['logg'].max(),self._instantaneous_gpole,self._instantaneous_rpole*self._scale))

    def _fill_grav(self, **kwargs):
        """
        TODO: add documentation

        requires _fill_logg to have been called
        """
        self._fill_column('grav', 0.0)

        #grav = abs(10**(self.mesh['logg']-2)/self._instantaneous_gpole)**self.gravb_bol

        #self._fill_column('grav', grav)

    def _fill_teff(self, **kwargs):
        """
        [NOT IMPLEMENTED]
        requires _fill_logg and _fill_grav to have been called



        """
        self._fill_column('teff', 0.0)


    def _fill_abun(self, abun=0.0):
        """
        TODO: add documentation
        """
        self._fill_column('abun', abun)


    def _populate_ifm(self, dataset, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`
        """
        raise NotImplementedError

    def _populate_rv(self, dataset, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`
        """

        # print "*** Star._populate_rv"
        passband = kwargs.get('passband', 'Johnson:V')
        ld_coeffs = kwargs.get('ld_coeffs', [0.5,0.5])
        ld_func = kwargs.get('ld_func', 'logarithmic')
        atm = kwargs.get('atm', 'kurucz')
        boosting_alg = kwargs.get('boosting_alg', 'none')

        # We need to fill all the flux-related columns so that we can weigh each
        # triangle's RV by its flux in the requested passband.
        lc_cols = self._populate_lc(dataset, **kwargs)

        # RV per element is just the z-component of the velocity vectory.  Note
        # the change in sign from our right-handed system to RV conventions.
        # These will be weighted by the fluxes when integrating
        rvs = -1*self.mesh['velo___bol_'][:,2]

        # Gravitational redshift
        if self.do_rv_grav:
            rv_grav = constants.G*(self.mass*u.solMass)/(self._instantaneous_rpole*u.solRad)/constants.c
            # rvs are in solrad/d internally
            rv_grav = rv_grav.to('solRad/d').value

            rvs += rv_grav

        cols = lc_cols
        cols['rv'] = rvs
        return cols


    def _populate_lc(self, dataset, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`

        :raises NotImplementedError: if lc_method is not supported
        """

        lc_method = kwargs.get('lc_method', 'numerical')  # TODO: make sure this is actually passed
        passband = kwargs.get('passband', 'Johnson:V')

        ld_coeffs = kwargs.get('ld_coeffs', [0.5,0.5])
        ld_func = kwargs.get('ld_func', 'logarithmic')
        atm = kwargs.get('atm', 'blackbody')
        boosting_alg = kwargs.get('boosting_alg', 'none')

        pblum = kwargs.get('pblum', 4*np.pi)

        mesh = self.mesh
        #mus = mesh['mu']

        if lc_method=='numerical':
            raise NotImplementedError


        elif lc_method=='analytical':
            raise NotImplementedError("analytical fluxes not yet ported to beta")

        else:
            raise NotImplementedError("lc_method '{}' not recognized".format(lc_method))


        # TODO: do we really need to store all of these if store_mesh==False?  Can we optimize by
        # only returning the essential if we know we don't need them?
        return {'intens_norm_abs': intens_norm_abs, 'intens_norm_rel': intens_norm_rel,
            'intens_proj_abs': intens_proj_abs, 'intens_proj_rel': intens_proj_rel,
            'ampl_boost': ampl_boost, 'ld': ld}
