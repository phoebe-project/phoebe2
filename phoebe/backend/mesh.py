import numpy as np
from math import sqrt, sin, cos, acos, atan2, trunc, pi
import copy

from phoebe import u
from phoebe import c
import libphoebe

import logging
logger = logging.getLogger("MESH")
logger.addHandler(logging.NullHandler())

def compute_volume(sizes, centers, normals):
    """
    Compute the numerical volume of a convex mesh

    :parameter array sizes: array of sizes of triangles
    :parameter array centers: array of centers of triangles (x,y,z)
    :parameter array normals: array of normals of triangles (will normalize if not already)
    :return: the volume (float)
    """
    # the volume of a slanted triangular cone is A_triangle * (r_vec dot norm_vec) / 3.

    # TODO: implement normalizing normals into meshing routines (or at least have them supply normal_mags to the mesh)


    # TODO: remove this function - should now be returned by the meshing algorithm itself
    # although wd method may currently use this
    normal_mags = np.linalg.norm(normals, axis=1) #np.sqrt((normals**2).sum(axis=1))
    return np.sum(sizes*((centers*normals).sum(axis=1)/normal_mags)/3)


def euler_trans_matrix(etheta, elongan, eincl):
    """
    Get the transformation matrix to translate/rotate a mesh according to
    euler angles

    :parameter float etheta: euler theta angle
    :parameter float elongan: euler long of asc node angle
    :parameter float eincl: euler inclination angle
    :return: matrix with size 3x3
    """

    s1 = sin(eincl);
    c1 = cos(eincl);
    s2 = sin(elongan);
    c2 = cos(elongan);
    s3 = sin(etheta);
    c3 = cos(etheta);
    c1s3 = c1*s3;
    c1c3 = c1*c3;

    return np.array([
                        [-c2*c3+s2*c1s3, c2*s3+s2*c1c3, -s2*s1],
                        [-s2*c3-c2*c1s3, s2*s3-c2*c1c3, c2*s1],
                        [s1*s3, s1*c3, c1]
                    ])

def transform_position_array(array, pos, euler, is_normal, reverse=False):
    """
    Transform any Nx3 position array by translating to a center-of-mass 'pos'
    and applying an euler transformation

    :parameter array array: numpy array of Nx3 positions in the original (star)
        coordinate frame
    :parameter array pos: numpy array with length 3 giving cartesian
        coordinates to offset all positions
    :parameter array euler: euler angles (etheta, elongan, eincl) in radians
    :parameter bool is_normal: whether each entry is a normal vector rather
        than position vector.  If true, the quantities won't be offset by
        'pos'
    :return: new positions array with same shape as 'array'.
    """
    trans_matrix = euler_trans_matrix(*euler)

    if not reverse:
        trans_matrix = trans_matrix.T

    if isinstance(array, ComputedColumn):
        array = array.for_computations

    if is_normal:
        # then we don't do an offset by the position
        return np.dot(np.asarray(array), trans_matrix)
    else:
        return np.dot(np.asarray(array), trans_matrix) + np.asarray(pos)

def transform_velocity_array(array, pos_array, vel, euler, rotation_vel=(0,0,0)):
    """
    Transform any Nx3 velocity vector array by adding the center-of-mass 'vel',
    accounting for solid-body rotation, and applying an euler transformation.

    :parameter array array: numpy array of Nx3 velocity vectors in the original
        (star) coordinate frame
    :parameter array pos_array: positions of the elements with respect to the
        original (star) coordinate frame.  Must be the same shape as 'array'.
    :parameter array vel: numpy array with length 3 giving cartesian velocity
        offsets in the new (system) coordinate frame
    :parameter array euler: euler angles (etheta, elongan, eincl) in radians
    :parameter array rotation_vel: vector of the rotation velocity of the star
        in the original (star) coordinate frame
    :return: new velocity array with same shape as 'array'
    """

    trans_matrix = euler_trans_matrix(*euler)

    rotation_component = np.cross(pos_array, rotation_vel, axisa=1)
    orbital_component = np.asarray(vel)

    if isinstance(array, ComputedColumn):
        array = array.for_computations

    new_vel = np.dot(np.asarray(array)+rotation_component, trans_matrix.T) + orbital_component

    return new_vel


def wd_grid_to_mesh_dict(the_grid, q, F, d):
    """
    Transform a wd-style mesh to the format used by PHOEBE. Namely this handles
    translating vertices from Nx9 to Nx3x3 and creating the array of indices
    for each triangle.

    :parameter record-array the_grid: output from discretize_wd_style
    :parameter float q: mass-ratio (M_this/M_sibling)
    :parameter float F: syncpar
    :parameter float d: instantaneous unitless separation
    :return: the dictionary in PHOEBE's format to be passed to a Mesh class
    """

    # WD returns a list of triangles with 9 coordinates (v1x, v1y, v1z, v2x, ...)
    triangles_9N = the_grid[:,4:13]

    new_mesh = {}
    # force the mesh to be computed at centers rather than the PHOEBE default
    # of computing at vertices and averaging for the centers.  This will
    # propogate to all ComputedColumns, which means we'll fill those quanities
    # (ie normgrads, velocities) per-triangle.
    new_mesh['compute_at_vertices'] = False
    # PHOEBE's mesh structure stores vertices in an Nx3 array
    new_mesh['vertices'] = triangles_9N.reshape(-1,3)
    # and triangles as indices pointing to each of the 3 vertices (Nx3)
    new_mesh['triangles'] = np.arange(len(triangles_9N)*3).reshape(-1,3)
    new_mesh['centers'] = the_grid[:,0:3]

    new_mesh['tnormals'] = the_grid[:,13:16]
    norms = np.linalg.norm(new_mesh['tnormals'], axis=1)
    new_mesh['normgrads'] = norms
    # TODO: do this the right way by dividing along axis=1 (or using np.newaxis as done for multiplying in ComputedColumns)
    new_mesh['tnormals'] = np.array([tn/n for tn,n in zip(new_mesh['tnormals'], norms)])

    # NOTE: there are no vnormals in wd-style mesh

    new_mesh['areas'] = the_grid[:,3]
    new_mesh['tareas'] = the_grid[:,18]

    # TESTING ONLY - remove this eventually ??? (currently being used
    # to test WD-style eclipse detection by using theta and phi (lat and long)
    # to determine which triangles are in the same "strip")
    new_mesh['thetas'] = the_grid[:,16]
    new_mesh['phis'] = the_grid[:,17]

    # TODO: get rid of this list comprehension
    # grads = np.array([libphoebe.roche_gradOmega_only(q, F, d, c) for c in new_mesh['centers']])

    # new_mesh['normgrads'] = np.sqrt(grads[:,0]**2+grads[:,1]**2+grads[:,2]**2)
    # new_mesh['normgrads'] = norms #np.linalg.norm(grads, axis=1)

    # TODO: actually compute the numerical volume (find old code)
    new_mesh['volume'] = compute_volume(new_mesh['areas'], new_mesh['centers'], new_mesh['tnormals'])
    new_mesh['velocities'] = np.zeros(new_mesh['centers'].shape)

    return new_mesh


class ComputedColumn(object):
    """
    Any non-geometric column in a Mesh should be a ComputedColumn.  Depending
    on the type of mesh (defined by mesh._compute_at_vertices), the computed
    column will either compute values at the vertices and return the weighted
    averages across a triangle when observing, OR compute at the centers and
    return that single value when observing.

    This class simply provides a single interface with logical/intuitive
    (hopefully) properties for setting and retrieving values for either
    type of mesh.
    """
    def __init__(self, mesh, value_for_computations=None, **kwargs):
        """
        TODO: add documentation
        """

        self._mesh = mesh

        # NOTE: it is ESSENTIAL that all of the following are np.array
        # and not lists... but is up to the user (phoebe backend since the
        # user will probably not dig this deep)
        self._vertices = None
        self._centers = None
        if value_for_computations is not None:
            self.set_for_computations(value_for_computations)

        self._compute_at_vertices = kwargs.get('compute_at_vertices', self.mesh._compute_at_vertices)

    @property
    def compute_at_vertices(self):
        return self._compute_at_vertices

    @property
    def mesh(self):
        """
        Access to the parent mesh

        :return: an instantiated :class:`Mesh`, :class:`ScaledProtoMesh`, or
            :class:`ProtoMesh` object.
        """
        return self._mesh

    @property
    def vertices(self):
        """
        Access to the quantities defined at the vertices, if applicable

        :return: numpy array
        """
        return self._vertices

    @property
    def vertices_per_triangle(self):
        """
        Access to the quantities defined at the vertices, but reshaped to be
        in the order of triangles, with three entries each

        :return: numpy array
        """
        if self.vertices is not None:
            return self.vertices[self.mesh.triangles]
        else:
            return None

    @property
    def centers(self):
        """
        Access to the quantities at the centers of each triangles.  In the
        case where the quantities are provided at the vertices, this will
        return the average of those values.

        :return: numpy array
        """
        if self.mesh._compute_at_vertices:
            return self.averages
        else:
            return self._centers

    @property
    def for_computations(self):
        """
        Access to the quantities as they should be used for computations
        (either .vertices or .centers depending on the value of
        mesh._compute_at_vertices)

        :return: numpy array
        """
        if self.mesh._compute_at_vertices:
            return self.vertices
        else:
            return self.centers

    @property
    def averages(self):
        """
        Access to the average of the values at the vertices for each triangle.
        If the quantities are defined at centers instead of vertices, this
        will return None.  Also see :method:`centers`.

        :return: numpy array or None
        """
        if not self.mesh._compute_at_vertices:
            return None

        return np.mean(self.vertices_per_triangle, axis=1)

    @property
    def weighted_averages(self):
        """
        Access to the weighted averages of the values at the vertices for each
        triangle based on the weights provided by mesh.weights.  This is most
        useful for partially visible triangles when using libphoebe's
        eclipse detection that returns weights for each vertex.

        Note that weights by default are set to 1/3 for each vertex, meaning
        that this will provide the same values as :meth:`averages` unless
        the weights are overridden within the mesh.

        If the quantities are defined at centers instead of vertices, this will
        return None.

        :return: numpy array or None
        """
        if not self.mesh._compute_at_vertices:
            return None

        vertices_per_triangle = self.vertices_per_triangle
        if vertices_per_triangle.ndim==2:
            # return np.dot(self.vertices_per_triangle, self.mesh.weights)
            return np.sum(vertices_per_triangle*self.mesh.weights, axis=1)
        elif vertices_per_triangle.ndim==3:
            return np.sum(vertices_per_triangle*self.mesh.weights[:,np.newaxis], axis=1)
        else:
            raise NotImplementedError

    @property
    def for_observations(self):
        """
        Access to the quantities as they should be used for observations.
        When defined at centers, this will return :meth:`centers`, when defined
        at vertices, this will return :meth:`weighted_averages`.

        :return: numpy array
        """
        if self.mesh._compute_at_vertices:
            # TODO: make this an option at some point?
            # return self.averages
            # for now this can just be achieved by calling self.centers directly
            return self.weighted_averages
        else:
            return self.centers

    def set_for_computations(self, value):
        """
        Set the quantities, either at the vertices or centers depending on the
        settings of the mesh (mesh._compute_at_vertices)
        """
        if self.mesh._compute_at_vertices:
            self._vertices = value
        else:
            self._centers = value


class ProtoMesh(object):
    """
    ProtoMesh is in units of d with the origin at the COM of the STAR.

    Because there is no orbital, scale, or orientation information, those
    fields are not available until the mesh is scaled and/or placed in
    orbit (by using the class constructors on ScaledProtoMesh or Mesh
    respectively).
    """

    def __init__(self, compute_at_vertices=True, **kwargs):
        """
        TODO: add documentation

        :parameter bool compute_at_vertices: whether all
            :class:`ComputedColumn` (physical quantities) should be computed
            at the vertices and then averaged or computed at the centers.
            Generally this should be decided based on whether the vertices
            or centers of the elements are placed on the surface of the
            equipotential.
        """

        self._compute_at_vertices = compute_at_vertices

        self._pvertices         = None  # Vx3
        self._vertices          = None  # Vx3

        self._triangles         = None  # Nx3
        self._centers           = None  # Nx3
        self._areas             = None  # Nx1
        self._tareas            = None  # Nx1

        self._velocities        = ComputedColumn(mesh=self)  # Nx1

        # v = vertices
        # t = triangle
        self._vnormals          = None  # Vx3
        self._tnormals          = None  # Nx3

        self._normgrads         = ComputedColumn(mesh=self)

        self._volume            = None  # scalar

        ### TESTING FOR WD METHOD ###
        self._phis               = None # Nx1
        self._thetas             = None # Nx1

        ### WD style OCs only ###
        self._env_comp            = None # Vx1
        self._env_comp3           = None # Nx1


        ### PHYSICAL QUANTITIES
        self._loggs             = ComputedColumn(mesh=self)
        self._gravs             = ComputedColumn(mesh=self)
        self._teffs             = ComputedColumn(mesh=self)
        self._abuns             = ComputedColumn(mesh=self)
        self._frac_refls          = ComputedColumn(mesh=self)
        # self._frac_heats          = ComputedColumn(mesh=self)
        # self._frac_scatts          = ComputedColumn(mesh=self)


        self._pos               = np.array([0.,0.,0.])  # will be updated when placed in orbit (only for Meshes)
        self._scalar_fields     = ['volume']

        if 'label_envelope' in kwargs.keys():
            self._label_envelope = kwargs.pop('label_envelope')
            self._label_primary = kwargs.pop('label_primary')
            self._label_secondary = kwargs.pop('label_secondary')


        # TODO: split keys that are set vs computed-on-the-fly so when
        # we call something like ScaledProtoMesh.from_proto we don't have
        # to do all the on-the-fly computations just to discard them because
        # they aren't setable.
        keys = ['pvertices', 'vertices', 'triangles', 'centers',
                  'coords_for_computations', 'normals_for_computations',
                  'rs', 'rprojs', 'cosbetas',
                  'areas', 'tareas', 'areas_si',
                  'velocities', 'vnormals', 'tnormals',
                  'normgrads', 'volume',
                  'phis', 'thetas', 'env_comp','env_comp3',
                  'compute_at_vertices',
                  'loggs', 'gravs', 'teffs', 'abuns', 'frac_refls'] # frac_heats, frac_scatts
        self._keys = keys + kwargs.pop('keys', [])

        self.update_columns(**kwargs)

    def __getitem__(self, key):
        """
        TODO: add documentation
        """

        # TODO: split this stuff into the correct classes/subclasses


        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self, '_observables') and key in self._observables.keys():
            # applicable only for Mesh, not ProtoMesh
            return self._observables[key]
        else:
            raise KeyError("{} is not a valid key".format(key))

    def __setitem__(self, key, value):
        """
        TODO: add documentation
        """

        # TODO: split this stuff into the correct classes/subclasses

        # NOTE: this will not be able to set values of ComputedColumns
        hkey = '_{}'.format(key)

        if hasattr(self, hkey):
            col = getattr(self, hkey)

            if isinstance(col, ComputedColumn) and not isinstance(value, ComputedColumn):
                col.set_for_computations(value)
            else:
                setattr(self, hkey, value)

            return

        elif hasattr(self, '_observables'):
            # applicable only for Mesh, not ProtoMesh
            # even if it doesn't exist, we'll make a new entry in observables]

            if key not in self._observables.keys():
                self._observables[key] = ComputedColumn(self)

            self._observables[key].set_for_computations(value)


        elif hasattr(self, key) and not hasattr(self, hkey):
            pass

        else:
            raise KeyError("{} is not a valid key".format(key))

    def keys(self):
        """
        TODO: add documentation
        """
        return self._keys

    def values(self):
        """
        TODO: add documentation
        """
        return self.items().values()

    def items(self):
        """
        TODO: add documentation
        """
        return {k: self[k] for k in self.keys()}

    def copy(self):
        """
        Make a deepcopy of this Mesh object
        """
        return copy.deepcopy(self)

    def update_columns_dict(self, kwargs):
        """
        Update the value of a column or multiple columns by passing as a dict.
        For observable columns, provide the label of the observable itself and
        it will be found (so long as it does not conflict with an existing
        non-observable column).
        """
        # make sure to do the geometric things that are needed for some of the
        # ComputedColumns first
        for key in ('triangles', 'vertices', 'centers', 'vnormals', 'tnormals'):
            if key in kwargs.keys():
                self.__setitem__(key, kwargs.pop(key))

        for k, v in kwargs.items():
            if isinstance(v, float) and k not in self._scalar_fields:
                # Then let's make an array with the correct length full of this
                # scalar

                # NOTE: this won't work for Nx3's, but that
                # really shouldn't ever happen since they should be set
                # within the init.
                # v = np.ones(self.Ntriangles)*v
                if self._compute_at_vertices:
                    v = np.full(self.Nvertices, v)
                else:
                    v = np.full(self.Ntriangles, v)

            if isinstance(v, ComputedColumn):
                # then let's update the mesh instance to correctly handle
                # inheritance
                v._mesh = self

            self.__setitem__(k, v)

    def update_columns(self, **kwargs):
        """
        Update the value of a column or multiple columns by passing as kwargs.
        For observable columns, provide the label of the observable itself and
        it will be found (so long as it does not conflict with an existing
        non-observable column).
        """
        self.update_columns_dict(kwargs)

    @property
    def compute_at_vertices(self):
        """
        Access (read-only) to the setting of whether computations should
        be done at the vertices (and then averaged) or at the centers
        of each triangle

        :return: bool
        """
        return self._compute_at_vertices

    @property
    def Ntriangles(self):
        """
        Return the number of TRIANGLES/ELEMENTS in the mesh.

        Simply a shortcut to len(self.triangles)
        """
        return len(self.triangles)

    @property
    def Nvertices(self):
        """
        Return the number of VERTICES in the mesh.

        Simply a shortcut to len(self.vertices)
        """
        return len(self.vertices)


    @property
    def pvertices(self):
        """
        Return the array of vertices on the potential, where each item is a
        triplet representing cartesian coordinates.

        (Vx3)
        """
        return self._pvertices

    @property
    def vertices(self):
        """
        Return the array of vertices, where each item is a triplet
        representing cartesian coordinates.

        (Vx3)
        """
        return self._vertices

    @property
    def pvertices_per_triangle(self):
        """
        TODO: add documentation

        TODO: confirm shape
        (Nx3x3)
        """
        return self.pvertices[self.triangles]


    @property
    def vertices_per_triangle(self):
        """
        TODO: add documentation

        TODO: confirm shape
        (Nx3x3)
        """
        return self.vertices[self.triangles]

    @property
    def triangles(self):
        """
        Return the array of triangles, where each item is a triplet of indices
        pointing to vertices.

        (Nx3)
        """
        return self._triangles

    @property
    def centers(self):
        """
        Return the array of centers, where each item is a triplet representing
        cartesian coordinates.

        (Nx3)
        """
        return self._centers

    @property
    def coords_for_observations(self):
        """
        Return the coordinates from the center of the star for each element
        (either centers or vertices depending on the setting in the mesh)
        after perturbations (either by features or by offsetting to get
        the correct volume).  NOTE: this is NOT necessarily where the physical
        parameters were computed, but IS where eclipse detection, etc, is
        handled.
        """
        if self._compute_at_vertices:
            return self.vertices - self._pos
        else:
            return self.centers - self._pos

    @property
    def coords_for_computations(self):
        """
        Return the coordinates from the center of the star for each element
        (either centers or vertices depending on the setting in the mesh).
        """

        # TODO: need to subtract the position offset if a Mesh (in orbit)
        if self._compute_at_vertices:
            if self.pvertices is not None:
                return self.pvertices - self._pos
            else:
                return self.vertices - self._pos
        else:
            return self.centers - self._pos

    @property
    def normals_for_computations(self):
        """
        Return the normals for each element
        (either centers or vertices depending on the setting in the mesh).
        """
        if self._compute_at_vertices:
            return self.vnormals
        else:
            return self.tnormals

    @property
    def rs(self):
        """
        Return the radius of each element (either vertices or centers
        depending on the setting in the mesh) with respect to the center of
        the star.

        (ComputedColumn)
        """
        rs = np.linalg.norm(self.coords_for_computations, axis=1)
        return ComputedColumn(self, rs)

    @property
    def rprojs(self):
        """
        Return the projected (in x,y plane) radius of each element (either
        vertices or centers depending on the setting in the mesh) with respect
        to the center of the star.

        (ComputedColumn)
        """
        # TODO: should this be moved to Mesh?  Even though its surely possible
        # to compute without being placed in orbit, projecting in x,y doesn't
        # make much sense without LOS orientation.
        rprojs = np.linalg.norm(self.coords_for_computations[:,:2], axis=1)
        return ComputedColumn(self, rprojs)

    @property
    def cosbetas(self):
        """
        TODO: add documentation

        (ComputedColumn)
        """
        coords = self.coords_for_computations
        norms = self.normals_for_computations

        # TODO: ditch the list comprehension... I know I figured out how to do
        # this (ie along an axis) with np.dot somewhere else
        cosbetas = np.array([np.dot(c,n) / (np.linalg.norm(c)*np.linalg.norm(n)) for c,n in zip(coords, norms)])

        return ComputedColumn(self, cosbetas)

    @property
    def areas(self):
        """
        Return the array of areas, where each item is a scalar/float.

        TODO: UNIT?

        (Nx1)
        """
        return self._areas

    @property
    def tareas(self):
        """
        Return the array of triangle areas, where each item is a scalar/float.

        TODO: UNIT?

        (Nx1)
        """
        return self._tareas

    @property
    def areas_si(self):
        """
        TODO: add documentation
        """
        if self._areas is not None:
            return (self.areas*u.solRad**2).to(u.m**2).value
        else:
            return None


    @property
    def velocities(self):
        """
        Return the array of velocities, where each item is a scalar/float.

        TODO: UNIT?

        (Nx1)
        """
        return self._velocities

    @property
    def vnormals(self):
        """
        TODO: add documentation

        (Vx3)
        """
        return self._vnormals


    @property
    def tnormals(self):
        """
        Return the array of tnormals (normals for each triangle face), where
        each items is a triplet representing a cartesian normaled vector.

        (Nx3)
        """
        return self._tnormals

    @property
    def normgrads(self):
        """
        TODO: add documentation

        (ComputedColumn)
        """
        return self._normgrads

    @property
    def volume(self):
        """
        Return the volume of the ENTIRE MESH.

        (scalar/float)
        """
        return self._volume

    @property
    def phis(self):
        """
        TODO: add documentation
        """
        # TODO: if self._phis is None then compute from cartesian
        return self._phis

    @property
    def thetas(self):
        """
        TODO: add documentation
        """
        # TODO: if self._thetas is None then compute from cartesian
        return self._thetas

    @property
    def env_comp(self):
        """
        TODO: add documentation
        """
        return self._env_comp

    @property
    def env_comp3(self):
        """
        TODO: add documentation
        """
        return self._env_comp3

    @property
    def loggs(self):
        """
        Return the array of loggs, where each item is a scalar/float.

        (ComputedColumn)
        """
        return self._loggs

    @property
    def gravs(self):
        """
        Return the array of gravs, where each item is a scalar/float.

        TODO: UNIT?

        (ComputedColumn)
        """
        return self._gravs

    @property
    def teffs(self):
        """
        Return the array of teffs, where each item is a scalar/float.

        Unit: K

        (ComputedColum)
        """
        return self._teffs

    @property
    def abuns(self):
        """
        Return the array of abuns, where each item is a scalar/float.

        TODO: UNIT?

        (ComputedColumn)
        """
        return self._abuns

    @property
    def frac_refls(self):
        """
        Return the array of frac_refls, where each item is a scalar/float

        (ComputedColumn)
        """
        return self._frac_refls

    # @property
    # def frac_heats(self):
    #     """
    #     Return the array of frac_heats, where each item is a scalar/float

    #     (ComputedColumn)
    #     """
    #     return self._frac_heats

    # @property
    # def frac_scatts(self):
    #     """
    #     Return the array of frac_scatts, where each item is a scalar/float

    #     (ComputedColumn)
    #     """
    #     return self._frac_scatts




class ScaledProtoMesh(ProtoMesh):
    """
    ScaledProtoMesh is in real units (in whatever is provided for scale),
    but still with the origin at the COM of the STAR.

    Because there is no orbital or orientation information, those
    fields are not available until the mesh is placed in
    orbit (by using the class constructor on Mesh).
    """

    def __init__(self, **kwargs):
        """
        TODO: add documentation
        """

        keys = []
        keys += kwargs.pop('keys', [])

        scale = kwargs.pop('scale', None)

        super(ScaledProtoMesh, self).__init__(keys=keys, **kwargs)

        if scale is not None:
            self._scale_mesh(scale)

    @classmethod
    def from_proto(cls, proto_mesh, scale):
        """
        TODO: add documentation
        """

        mesh = cls(**proto_mesh.items())
        mesh._scale_mesh(scale=scale)

        if hasattr(proto_mesh, '_label_envelope'):
            mesh._label_envelope = proto_mesh._label_envelope
            mesh._label_primary = proto_mesh._label_primary
            mesh._label_secondary = proto_mesh._label_secondary

        return mesh

    def _scale_mesh(self, scale):
        """
        TODO: add documentation
        """
        pos_ks = ['vertices', 'centers']

        # TODO: scale velocities???

        # handle scale
        self.update_columns_dict({k: self[k]*scale for k in pos_ks})

        self.update_columns(areas=self.areas*(scale**2))
        self._volume *= scale**3
        # TODO NOW: scale volume



class Mesh(ScaledProtoMesh):
    """
    Mesh is in real units (in whatever is provided for scale) with the
    origin at the COM of the system (the coordinate system that should be
    used when passing pos, vel, euler).

    Now that the mesh is in orbit and has viewing orientation information,
    observables can be attached.
    """
    def __init__(self, **kwargs):

        # self._mus               = None  # Nx1 property computed on-the-fly
        self._visibilities      = None  # Nx1
        self._weights           = None  # Nx3 (per vertex, same order as self.triangles)

        self._observables       = {}    # ComputedColumn (each)

        keys = ['mus', 'visibilities', 'weights', 'observables']
        keys = keys + kwargs.pop('keys', [])

        super(Mesh, self).__init__(keys=keys, **kwargs)

    @classmethod
    def from_proto(cls, proto_mesh, scale,
                      pos, vel, euler, rotation_vel=(0,0,0)):
        """
        Turn a ProtoMesh into a Mesh scaled and placed in orbit.

        Update all geometry fields from the proto reference frame, to the
        current system reference frame, given the current position, velocitiy,
        euler angles, and rotational velocity of THIS mesh.

        :parameter list pos: current position (x, y, z)
        :parameter list vel: current velocity (vx, vy, vz)
        :parameter list euler: current euler angles (etheta, elongan, eincl)
        :parameter list rotation_vel: rotation velocity vector (polar_dir*freq_rot)
        """

        mesh = cls(**proto_mesh.items())

        mesh._scale_mesh(scale=scale)
        mesh._place_in_orbit(pos, vel, euler, rotation_vel)

        if hasattr(proto_mesh, '_label_envelope'):
            mesh._label_envelope = proto_mesh._label_envelope
            mesh._label_primary = proto_mesh._label_primary
            mesh._label_secondary = proto_mesh._label_secondary

        return mesh

    @classmethod
    def from_scaledproto(cls, scaledproto_mesh,
                         pos, vel, euler, rotation_vel=(0,0,0)):
        """
        TODO: add documentation
        """

        mesh = cls(**scaledproto_mesh.items())

        mesh._place_in_orbit(pos, vel, euler, rotation_vel)

        if hasattr(scaledproto_mesh, '_label_envelope'):
            mesh._label_envelope = scaledproto_mesh._label_envelope
            mesh._label_primary = scaledproto_mesh._label_primary
            mesh._label_secondary = scaledproto_mesh._label_secondary

        return mesh

    def _place_in_orbit(self, pos, vel, euler, rotation_vel=(0,0,0)):
        """
        TODO: add documentation
        """

        # TODO: store pos, vel, euler so that INCREMENTAL changes are allowed
        # if passing new values (and then make this a public method).  See note
        # below!

        pos_ks = ['vertices', 'pvertices', 'centers']
        norm_ks = ['vnormals', 'tnormals'] #, 'cnormals']
        vel_ks = ['velocities']

        # NOTE: we do velocities first since they require the positions WRT
        # the star (not WRT the system).  Will need to keep this in mind if we
        # eventually support incremental transformations.
        pos_array = self.vertices if self._compute_at_vertices else self.centers
        self.update_columns_dict({k: transform_velocity_array(self[k], pos_array, vel, euler, rotation_vel) for k in vel_ks if self[k] is not None})
        # TODO: handle velocity from mesh reprojection during volume conservation

        # handle rotation/displacement
        # NOTE: mus will automatically be updated on-the-fly
        self.update_columns_dict({k: transform_position_array(self[k], pos, euler, False) for k in pos_ks if self[k] is not None})
        self.update_columns_dict({k: transform_position_array(self[k], pos, euler, True) for k in norm_ks if self[k] is not None})

        # let's store the position.  This is both useful for "undoing" the
        # orbit-offset, and also eventually to allow incremental changes.
        self._pos = pos


    @property
    def mus(self):
        """
        Return the array of mus (z-coordinate of tnormals), where each item
        is a scalar/float.

        (Nx1)
        """
        # this requires tnormals to be NORMALIZED (currently we don't do any
        # checks - this is just assumed)
        return self.tnormals[:,2]

    @property
    def mus_for_computations(self):
        """
        TODO: add documentation

        (Nx1 or Vx1)
        """
        if self._compute_at_vertices:
            # this requires vnormals to be NORMALIZED (assumed)
            return self.vnormals[:,2]
        else:
            # this is just self.tnormals[:,2] from above
            return self.mus

    @property
    def visibilities(self):
        """
        Return the array of visibilities, where each item is a scalar/float
        between 0 (completely hidden/invisible) and 1 (completely visible).

        (Nx1)
        """
        if self._visibilities is not None:
            return self._visibilities
        else:
            return np.ones(self.Ntriangles)

    @property
    def weights(self):
        """
        TODO: add documentation

        (Nx3)
        """
        if self._weights is not None and len(self._weights):
            return self._weights
        else:
            return np.full((self.Ntriangles, 3), 1./3)

    @property
    def observables(self):
        """
        Return the dictionary of observables

        (ComputedColumn)
        """
        return self._observables

    def get_observable(self, label):
        """
        Retrieve the array of an observable by its label.

        (ComputedColumn)
        """
        return self.observables[label]

    def fill_observable(self, label, value):
        """
        Fill the array of an observable by its label.  The provided value
        MUST be either a scalar (applied for all triangles), or an array
        with length N.  This IS NOT checked, but failing to provide the
        correct shape could have dire consequences.
        """
        self._observables[label] = value

    def update_columns_dict(self, kwargs):
        """
        TODO: add documentation
        """
        super(Mesh, self).update_columns_dict(kwargs)

        # if kwargs.get('vnormals', None) is not None or kwargs.get('tnormals', None) is not None:
            # self._compute_mus()
        if kwargs.get('triangles', None) is not None:
            # reset visibilities and velocities so that they are reset
            # when next queried
            self.update_columns(visibilities=None, velocities=None)


class Meshes(object):
    """
    This is just a class to allow easy access to meshes at the system-level.
    This allows eclipse detection and subdivision to work at the system-level in
    an efficient and clean way.

    In effect this has the same structure as the system, but allows passing only
    the meshes and dropping all the Body methods/attributes.
    """
    def __init__(self, items, parent_envelope_of={}):
        """
        TODO: add documentation

        :parameter list items:
        """
        self._dict = {component: body.mesh for component, body in items.items()}
        #self._component_by_no = {body.comp_no: component for component, body in items.items()}
        self._components = items.keys()
        self._parent_envelope_of = parent_envelope_of

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
        TODO: add documentation
        """
        return self._dict[key]

    def __setitem__(self, key, value):
        """
        TODO: add documentation
        """
        return self.update_columns(key, value)

    def component_by_no(self, comp_no):
        """
        TODO: add documentation
        """
        #return self._component_by_no[comp_no]
        return self._components[comp_no-1]

    def update_columns(self, field, value_dict, inds=None, computed_type=None):
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
            if computed_type is not None:
                # then create the ComputedColumn now to override the default value of compute_at_vertices
                self._dict[comp]._observables[field] = ComputedColumn(self._dict[comp], compute_at_vertices=computed_type=='vertices')

            #print "***", comp, field, inds, value
            if inds:
                raise NotImplementedError('setting column with indices not yet ported to new meshing')
                # self._dict[comp][field][inds] = value
            else:
                self._dict[comp][field] = value

    def get_column(self, field, components=None, computed_type='for_observations'):
        """
        TODO: add documentation

        return a dictionary for a single column, with component as keys and the
        column array as values

        :parameter str field: name of the mesh columnname
        :parameter components:
        """
        def get_field(c, field, computed_type):
            if c not in self._dict.keys():
                # then handle the case where we're requesting a star in an
                # envelope (ie. request the "primary" half of "contact_envelope")

                c_orig = c
                c = self._parent_envelope_of[c]
                # TODO: how in the world do we access this logic????????
                # right now its stored in Envelope.label_primary
                if c_orig == self._dict[c]._label_primary:
                    comp_no = 0
                elif c_orig == self._dict[c]._label_secondary:
                    comp_no = 1
                else:
                    raise ValueError
            else:
                comp_no = None

            f = self._dict[c][field]
            if isinstance(f, ComputedColumn):
                col = getattr(f, computed_type)
            else:
                col =  f

            if comp_no is None:
                col = col
            elif comp_no == 0:
                col = col[self._dict[c]['env_comp3'] == 0]
            elif comp_no == 1:
                col = col[self._dict[c]['env_comp3'] == 1]
            else:
                raise NotImplementedError

            return col

        if components:
            if isinstance(components, str):
                components = [components]
        else:
            components = self.keys()

        return {c: get_field(c, field, computed_type) for c in components}

    def get_column_flat(self, field, components=None, computed_type='for_observations'):
        """
        TODO: add documentation

        return a single merged value (hstacked) from all meshes

        :parameter str field: name of the mesh columnname
        :parameter components:
        """
        return self.pack_column_flat(self.get_column(field, components, computed_type),
                                     components,
                                     offset=field=='triangles')

    def pack_column_flat(self, value, components=None, offset=False):
        """
        TODO: add documentation
        """
        if components:
            if isinstance(components, str):
                components = [components]
        elif isinstance(value, dict):
            components = value.keys()
        elif isinstance(value, list):
            components = self._dict.keys()
            value = {c: v for c,v in zip(components, value)}

        if offset:
            values = []
            offsetN = 0
            for c in components:
                values.append(value[c]+offsetN)
                offsetN += len(self[c]['vertices'])

        else:
            values = [value[c] for c in components]

        if len(value[components[0]].shape) > 1:
            return np.vstack(values)
        else:
            return np.hstack(values)

    def unpack_column_flat(self, value, components=None, offset=False, computed_type=None):
        """
        TODO: add documentation
        TODO: needs testing
        """
        if components:
            if isinstance(components, str):
                components = [components]
        else:
            components = self._dict.keys()

            # TODO: add this

        # we need to split the flat array by the lengths of each mesh
        N_lower = 0
        N_upper = 0
        offsetN = 0.0
        value_dict = {}
        for c in components:
            mesh = self[c]
            if computed_type=='vertices' or (computed_type is None and mesh._compute_at_vertices):
                N = mesh.Nvertices
            else:
                N = mesh.Ntriangles
            N_upper += N
            value_dict[c] = value[N_lower:N_upper] - offsetN
            if offset:
                offsetN += N
            N_lower += N

        return value_dict

    def set_column_flat(self, field, value, components=None, computed_type=None):
        """
        TODO: add documentation
        TODO: needs testing
        """
        value_dict = self.unpack_column_flat(value, components, computed_type=computed_type)
        self.update_columns(field, value_dict, computed_type=computed_type)

    def replace_elements(self, inds, new_submesh, component):
        """
        TODO: add documentation
        TODO: remove this method???
        """
        self._dict[component] = np.hstack([self._dict[component][~inds], new_submesh])
