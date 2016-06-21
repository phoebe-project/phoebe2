import numpy as np
from math import sqrt, sin, cos, acos, atan2, trunc, pi
import copy

import libphoebe

import logging
logger = logging.getLogger("MESH")
logger.addHandler(logging.NullHandler())

def compute_volume(sizes, centers, normals):
    """
    Compute the numerical volume of a convex

    :parameter array sizes: array of sizes of triangles
    :parameter array centers: array of centers of triangles (x,y,z)
    :parameter array normals: array of normals of triangles (will normalize if not already)
    """
    # the volume of a slanted triangular cone is A_triangle * (r_vec dot norm_vec) / 3.

    # TODO: implement normalizing normals into meshing routines (or at least have them supply normal_mags to the mesh)


    # TODO: remove this function - should now be returned by the meshing algorithm itself
    normal_mags = np.sqrt((normals**2).sum(axis=1))
    return np.sum(sizes*((centers*normals).sum(axis=1)/normal_mags)/3)


def euler_trans_matrix(etheta, elongan, eincl):
    """
    # TODO: add documentation
    """

    # print "***", eincl
    # eincl = 0.0

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

def transform_position_array(array, pos, euler, is_normal):
    """
    TODO: add documentation

    pos is center of mass position (in new system frame)
    """
    trans_matrix = euler_trans_matrix(*euler)

    if isinstance(array, ComputedColumn):
        array = array.for_computations

    if is_normal:
        # then we don't do an offset by the position
        return np.dot(np.asarray(array), trans_matrix.T)
    else:
        return np.dot(np.asarray(array), trans_matrix.T) + np.asarray(pos)

def transform_velocity_array(array, vel, euler, rotation_vel=(0,0,0)):
    """
    TODO: add documentation

    vel is center of mass velocitiy (in new system frame)
    rotation_vel is vector of the rotation velocity of the star (in original frame?)
    """
    trans_matrix = np.cross(euler_trans_matrix(*euler), np.asarray(rotation_vel))



    if isinstance(array, ComputedColumn):
        array = array.for_computations

    new_vel = np.dot(np.asarray(array), trans_matrix.T) + np.asarray(vel)

    return new_vel


def wd_grid_to_mesh_dict(the_grid, q, F, d):
    # WD returns a list of triangles with 9 coordinates (v1x, v1y, v1z, v2x, ...)
    triangles_9N = the_grid[:,4:13]

    new_mesh = {}
    # force the mesh to be computed at centers rather than the PHOEBE default
    # of computing at vertices and averaging for the centers.  This will
    # propogate to all ComputedColumns
    new_mesh['compute_at_vertices'] = False
    # PHOEBE's mesh structure stores vertices in an Nx3 array
    new_mesh['vertices'] = triangles_9N.reshape(-1,3)
    # and triangles as indices pointing to each of the 3 vertices (Nx3)
    new_mesh['triangles'] = np.arange(len(triangles_9N)*3).reshape(-1,3)
    new_mesh['centers'] = the_grid[:,0:3]
    # NOTE: we're setting tnormals here because they're the ones that are
    # currently used to compute everything.
    new_mesh['tnormals'] = the_grid[:,13:16]
    norms = np.linalg.norm(new_mesh['tnormals'], axis=1)
    # TODO: do this the right way by dividing along axis=1
    new_mesh['tnormals'] = np.array([tn/n for tn,n in zip(new_mesh['tnormals'], norms)])

    new_mesh['areas'] = the_grid[:,3]

    # TESTING ONLY - remove this eventually ??? (currently being used
    # to test WD-style eclipse detection by using theta and phi (lat and long)
    # to determine which triangles are in the same "strip")
    new_mesh['theta'] = the_grid[:,16]
    new_mesh['phi'] = the_grid[:,17]

    grads = np.array([libphoebe.roche_gradOmega_only(q, F, d, c) for c in new_mesh['centers']])
    new_mesh['normgrads'] = np.sqrt(grads[:,0]**2+grads[:,1]**2+grads[:,2]**2)

    # TODO: actually compute the numerical volume (find old code)
    new_mesh['volume'] = compute_volume(new_mesh['areas'], new_mesh['centers'], new_mesh['tnormals'])
    new_mesh['velocities'] = np.zeros(new_mesh['centers'].shape)



    return new_mesh


class ComputedColumn(object):
    def __init__(self, mesh, **kwargs):

        self._mesh = mesh

        # NOTE: it is ESSENTIAL that all of the following are np.array
        # and not lists... but is up to the user (phoebe backend since the
        # user will probably not dig this deep)

        # NOTE: only one of these two arrays should really be filled based on
        # the value of self.mesh._compute_at_vertices
        self._vertices = kwargs.get('vertices', None) # N*3x1
        self._centers = kwargs.get('centers', None) # Nx1

    @property
    def mesh(self):
        return self._mesh

    @property
    def vertices(self):
        return self._vertices

    @property
    def vertices_per_triangle(self):
        if self.vertices is not None:
            return self.vertices[self.mesh.triangles]
        else:
            return None

    @property
    def centers(self):
        if self.mesh._compute_at_vertices:
            return self.averages
        else:
            return self._centers

    @property
    def for_computations(self):
        if self.mesh._compute_at_vertices:
            return self.vertices
        else:
            return self.centers

    @property
    def averages(self):
        if not self.mesh._compute_at_vertices:
            return None

        return np.mean(self.vertices_per_triangle, axis=1)

    @property
    def weighted_averages(self):
        if not self.mesh._compute_at_vertices:
            return None

        vertices_per_triangle = self.vertices_per_triangle
        if vertices_per_triangle.ndim==2:
            # return np.dot(self.vertices_per_triangle, self.weights)
            return np.sum(vertices_per_triangle*self.mesh.weights, axis=1)
        elif vertices_per_triangle.ndim==3:
            # np.sum(self.mesh.velocities.vertices_per_triangle[:,:,0]*self.mesh.velocities.mesh.weights, axis=1)
            # np.sum(self.mesh.velocities.vertices_per_triangle[:,:,1]*self.mesh.velocities.mesh.weights, axis=1)
            # np.sum(self.mesh.velocities.vertices_per_triangle[:,:,2]*self.mesh.velocities.mesh.weights, axis=1)
            return np.sum(vertices_per_triangle*self.mesh.weights[:,np.newaxis], axis=1)
        else:
            raise NotImplementedError


    @property
    def for_observations(self):
        if self.mesh._compute_at_vertices:
            return self.weighted_averages
        else:
            return self.centers



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
        """


        self._vertices          = None  # N*3x3

        self._triangles         = None  # Nx3
        self._centers           = None  # Nx3
        self._areas             = None  # Nx1

        # TODO: should velocities be a ComputedColumn???
        self._velocities        = ComputedColumn(mesh=self)  # Nx1

        # v = vertices
        # t = triangle
        # c = center (reprojected to surface)
        # a = average (average of values at each vertex)
        # av = average visibile (weighted average based on the visible portion)
        self._vnormals          = None  # N*3x3
        self._tnormals          = None  # Nx3
        # self._cnormals          = None  # Nx3

        # self._normals           = ComputedColumn(mesh=self)
        self._normgrads         = ComputedColumn(mesh=self)

        # TODO: ditch these in the future
        # self._cnormgrads        = None  # Nx1

        # self._vnormgrads        = None  # N*3x1
        # self._anormgrads        = None  # Nx1
        # self._avnormgrads       = None

        self._volume            = None  # scalar


        ### TESTING FOR WD METHOD ###
        self._phi               = None # Nx1
        self._theta             = None # Nx1

        self._scalar_fields     = ['volume']
        self._compute_at_vertices = compute_at_vertices

        keys = ['vertices', 'triangles', 'centers', 'areas',
                      'velocities', 'vnormals', 'tnormals',
                      'normgrads', 'volume',
                      'phi', 'theta',
                      'compute_at_vertices']
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
            # TODO NOW: this will break for computedcolumns... :-(
            return self._observables[key]
        else:
            raise KeyError

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
                if self._compute_at_vertices:
                    col._vertices = value
                else:
                    col._centers = value
            else:
                setattr(self, hkey, value)

            return

        elif hasattr(self, '_observables'):
            # applicable only for Mesh, not ProtoMesh
            # even if it doesn't exist, we'll make a new entry in observables]

            self._observables[key] = value

        else:
            raise KeyError("{} is not a valid key".format(key))

    def keys(self):
        return self._keys

    def values(self):
        return self.items().values()

    def items(self):
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
        # make sure to do triangles first, since that is needed for any of the
        # ComputedColumns
        if 'triangles' in kwargs.keys():
            self.__setitem__('triangles', kwargs.pop('triangles'))

        for k, v in kwargs.items():
            if isinstance(v, float) and k not in self._scalar_fields:
                # Then let's make an array with the correct length full of this
                # scalar

                # NOTE: this won't work for vertices, but that really shouldn't
                # ever happen
                v = np.ones(self.N)*v

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
        return self._compute_at_vertices


    @property
    def N(self):
        """
        Return the number of TRIANGLES/ELEMENTS in the mesh.

        Simply a shortcut to len(self.triangles)
        """
        return len(self.triangles)

    @property
    def vertices(self):
        """
        Return the array of verticles, where each item is a triplet
        representing cartesian coordinates.

        (N*3x3)
        """
        return self._vertices

    @property
    def vertices_per_triangle(self):
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
    def areas(self):
        """
        Return the array of areas, where each item is a scalar/float.

        TODO: UNIT?

        (Nx1)
        """
        return self._areas

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
        return self._vnormals


    # @property
    # def vnormals(self):
    #     """
    #     Return the array of vnormals (normals at each vertex), where each item
    #     is a triplet representing a cartesian normaled vector.

    #     (N*3x3)
    #     """
    #     return self._vnormals

    @property
    def tnormals(self):
        """
        Return the array of tnormals (normals for each triangle face), where
        each items is a triplet representing a cartesian normaled vector.

        (Nx3)
        """
        return self._tnormals

    # @property
    # def cnormals(self):
    #     """
    #     Return the array of cnormals (normals for SURFACE at the center of
    #     each triangle), where each items is a triplet representing a cartesian
    #     normaled vector.

    #     (Nx3)
    #     """
    #     return self._cnormals

    @property
    def normgrads(self):
        # NOTE: must use self._normgrads to access the ComputedColumn
        return self._normgrads #.weighted_averages


    # @property
    # def cnormgrads(self):
    #     """
    #     Return the array of cnormgrads (normals of the gradients for SURFACE
    #     at the center of each triangle), where each item is a scalar/float

    #     (Nx1)
    #     """
    #     return self._cnormgrads

    @property
    def volume(self):
        """
        Return the volume of the ENTIRE MESH.

        (scalar/float)
        """
        return self._volume

    @property
    def phi(self):
        return self._phi

    @property
    def theta(self):
        return self._theta



class ScaledProtoMesh(ProtoMesh):
    """
    ScaledProtoMesh is in real units (in whatever is provided for scale),
    but still with the origin at the COM of the STAR.

    Because there is no orbital or orientation information, those
    fields are not available until the mesh is placed in
    orbit (by using the class constructor on Mesh).
    """

    def __init__(self, **kwargs):

        self._loggs             = ComputedColumn(mesh=self)
        self._gravs             = ComputedColumn(mesh=self)
        self._teffs             = ComputedColumn(mesh=self)
        self._abuns             = ComputedColumn(mesh=self)

        keys = ['loggs', 'gravs', 'teffs', 'abuns']
        keys += kwargs.pop('keys', [])

        super(ScaledProtoMesh, self).__init__(keys=keys, **kwargs)

    @classmethod
    def from_proto(cls, proto_mesh, scale):
        """
        TODO: add documentation
        """

        mesh = cls(**proto_mesh.items())
        mesh._scale_mesh(scale=scale)
        return mesh

    def _scale_mesh(self, scale):
        pos_ks = ['vertices', 'centers']

        # handle scale
        self.update_columns_dict({k: self[k]*scale for k in pos_ks})

        self.update_columns(areas=self.areas*scale**2)
        # TODO NOW: scale volume

    @property
    def loggs(self):
        """
        Return the array of loggs, where each item is a scalar/float.

        (Nx1)
        """
        return self._loggs #.weighted_averages

    @property
    def gravs(self):
        """
        Return the array of gravs, where each item is a scalar/float.

        TODO: UNIT?

        (Nx1)
        """
        return self._gravs #.weighted_averages

    @property
    def teffs(self):
        """
        Return the array of teffs, where each item is a scalar/float.

        Unit: K

        (Nx1)
        """
        return self._teffs #.weighted_averages

    @property
    def abuns(self):
        """
        Return the array of abuns, where each item is a scalar/float.

        TODO: UNIT?

        (Nx1)
        """
        return self._abuns #.weighted_averages



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

        # TODO: should observables be computed from teffs.weighted_averages, etc
        # or itself be ComputedColumns??
        self._observables       = {}    # Nx1 (each)

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

        return mesh

    @classmethod
    def from_scaledproto(cls, scaledproto_mesh,
                            pos, vel, euler, rotation_vel=(0,0,0)):
        """
        TODO: add documentation
        """

        mesh = cls(**scaledproto_mesh.items())

        mesh._place_in_orbit(pos, vel, euler, rotation_vel)

        return mesh

    def _place_in_orbit(self, pos, vel, euler, rotation_vel=(0,0,0)):
        """
        TODO: add documentation
        """

        # TODO: store pos, vel, euler so that INCREMENTAL changes are allowed
        # if passing new values (and then make this a public method)

        pos_ks = ['vertices', 'centers']
        norm_ks = ['vnormals', 'tnormals'] #, 'cnormals']
        vel_ks = ['velocities']


        # handle rotation/displacement
        # NOTE: mus will automatically be updated when updating normals
        self.update_columns_dict({k: transform_position_array(self[k], pos, euler, False) for k in pos_ks if self[k] is not None})
        self.update_columns_dict({k: transform_position_array(self[k], pos, euler, True) for k in norm_ks if self[k] is not None})
        self.update_columns_dict({k: transform_velocity_array(self[k], vel, euler, rotation_vel) for k in vel_ks if self[k] is not None})
        # TODO: handle velocity from mesh reprojection during volume conservation


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
    def visibilities(self):
        """
        Return the array of visibilities, where each item is a scalar/float
        between 0 (completely hidden/invisible) and 1 (completely visible).

        (Nx1)
        """
        if self._visibilities is not None:
            return self._visibilities
        else:
            return np.ones(self.N)

    @property
    def weights(self):
        if self._weights is not None:
            return self._weights
        else:
            return np.full((self.N, 3), 1./3)


    @property
    def observables(self):
        """
        Return the dictionary of observables, where each entry is an array
        with shape Nx1
        """
        return self._observables

    def get_observable(self, label):
        """
        Retrieve the array of an observable by its label.
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
            # when nexted queried
            self.update_columns(visibilities=None, velocities=None)




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
        TODO: add documentation
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
            f = self._dict[c][field]
            if isinstance(f, ComputedColumn):
                return getattr(f, computed_type)
            else:
                return f

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
        else:
            components = value.keys()

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

    def unpack_column_flat(self, value, components=None, offset=False):
        """
        TODO: add documentation
        TODO: needs testing
        """
        if components:
            if isinstance(components, str):
                components = [components]
        else:
            components = self._dict.keys()

        # we need to split the flat array by the lengths of each mesh
        N_lower = 0
        N_upper = 0
        offsetN = 0.0
        value_dict = {}
        for c in components:
            mesh = self[c]
            N = mesh.N
            N_upper += N
            value_dict[c] = value[N_lower:N_upper] - offsetN
            if offset:
                offsetN += N
            N_lower += N

        return value_dict

    def set_column_flat(self, field, value, components=None):
        """
        TODO: add documentation
        TODO: needs testing
        """
        value_dict = self.unpack_column_flat(value, components)
        self.update_columns(field, value_dict)

    def replace_elements(self, inds, new_submesh, component):
        """
        TODO: add documentation
        TODO: remove this method???
        """
        self._dict[component] = np.hstack([self._dict[component][~inds], new_submesh])