import numpy as np
from scipy.optimize import newton
from scipy.special import sph_harm as Y
from math import sqrt, sin, cos, acos, atan2, trunc, pi
import os
import copy

from phoebe.atmospheres import passbands
from phoebe.distortions import roche, rotstar
from phoebe.backend import eclipse, potentials, mesh
import libphoebe

from phoebe import u
from phoebe import c
from phoebe import conf

import logging
logger = logging.getLogger("UNIVERSE")
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
        return [_value(o) for o in obj]
    return obj


class System(object):
    def __init__(self, bodies_dict, eclipse_method='graham',
                 horizon_method='boolean',
                 dynamics_method='keplerian',
                 irrad_method='none',
                 boosting_method='none',
                 parent_envelope_of={}):
        """
        :parameter dict bodies_dict: dictionary of component names and Bodies (or subclass of Body)
        """
        self._bodies = bodies_dict
        self._parent_envelope_of = parent_envelope_of
        self.eclipse_method = eclipse_method
        self.horizon_method = horizon_method
        self.dynamics_method = dynamics_method
        self.irrad_method = irrad_method
        for body in self._bodies.values():
            body.boosting_method = boosting_method

        return

    def copy(self):
        """
        Make a deepcopy of this Mesh object
        """
        return copy.deepcopy(self)

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
                compute_ps = b.get_compute(compute, check_visible=False)
            else:
                # then hopefully compute is the parameterset
                compute_ps = compute
            eclipse_method = compute_ps.get_value(qualifier='eclipse_method', **kwargs)
            horizon_method = compute_ps.get_value(qualifier='horizon_method', check_visible=False, **kwargs)
            # subdiv_alg = 'edge' #compute_ps.get_value(qualifier='subdiv_alg', **kwargs)
            # subdiv_num = compute_ps.get_value(qualifier='subdiv_num', **kwargs)
            dynamics_method = compute_ps.get_value(qualifier='dynamics_method', **kwargs)
            irrad_method = compute_ps.get_value(qualifier='irrad_method', **kwargs)
            boosting_method = compute_ps.get_value(qualifier='boosting_method', **kwargs)
            if conf.devel:
                mesh_init_phi = compute_ps.get_value(qualifier='mesh_init_phi', unit=u.rad, **kwargs)
            else:
                mesh_init_phi = 0.0
        else:
            eclipse_method = 'native'
            horizon_method = 'boolean'
            # subdiv_alg = 'edge'
            # subdiv_num = 3
            dynamics_method = 'keplerian'
            irrad_method = 'none'
            boosting_method = 'none'
            mesh_init_phi = 0.0

        # NOTE: here we use globals()[Classname] because getattr doesn't work in
        # the current module - now this doesn't really make sense since we only
        # support stars, but eventually the classname could be Disk, Spot, etc
        if 'dynamics_method' in kwargs.keys():
            # already set as default above
            _dump = kwargs.pop('dynamics_method')

        #starrefs  = hier.get_stars()
        #bodies_dict = {star: globals()['Star'].from_bundle(b, star, compute, dynamics_method=dynamics_method, datasets=datasets, **kwargs) for star in starrefs}

        meshables = hier.get_meshables()
        bodies_dict = {comp: globals()[hier.get_kind_of(comp).title()].from_bundle(b, comp, compute, dynamics_method=dynamics_method, mesh_init_phi=mesh_init_phi, datasets=datasets, **kwargs) for comp in meshables}

        # envelopes need to know their relationships with the underlying stars
        parent_envelope_of = {}
        for meshable in meshables:
            if hier.get_kind_of(meshable) == 'envelope':
                for starref in hier.get_siblings_of(meshable):
                    parent_envelope_of[starref] = meshable

        return cls(bodies_dict, eclipse_method=eclipse_method,
                   horizon_method=horizon_method,
                   dynamics_method=dynamics_method,
                   irrad_method=irrad_method,
                   boosting_method=boosting_method,
                   parent_envelope_of=parent_envelope_of)

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
        if component in self._bodies.keys():
            return self._bodies[component]
        else:
            # then hopefully we're a child star of an contact_binary envelope
            parent_component = self._parent_envelope_of[component]
            return self._bodies[parent_component]

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
        # meshes.update_column(visibilities=visibilities)
        # meshes.update_column(somethingelse=somethingelse)
        #
        # rather than calling self.meshes repeatedly

        return mesh.Meshes(self._bodies, self._parent_envelope_of)


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


    def update_positions(self, time, xs, ys, zs, vxs, vys, vzs,
                         ethetas, elongans, eincls,
                         ds=None, Fs=None, ignore_effects=False):
        """
        TODO: add documentation

        all arrays should be for the current time, but iterable over all bodies
        """
        self.xs = np.array(_value(xs))
        self.ys = np.array(_value(ys))
        self.zs = np.array(_value(zs))

        for starref,body in self.items():
            #logger.debug("updating position of mesh for {}".format(starref))
            body.update_position(time, xs, ys, zs, vxs, vys, vzs,
                                 ethetas, elongans, eincls,
                                 ds=ds, Fs=Fs, ignore_effects=ignore_effects)


    def populate_observables(self, time, kinds, datasets, ignore_effects=False):
        """
        TODO: add documentation

        ignore_effects: whether to ignore reflection and features (useful for computing luminosities)
        """


        if self.irrad_method is not 'none' and not ignore_effects:  # and kinds includes a kind that requires fluxes
            self.handle_reflection()

        for kind, dataset in zip(kinds, datasets):
            for starref, body in self.items():
                body.populate_observable(time, kind, dataset)

    def handle_reflection(self,  **kwargs):
        """
        """
        if self.irrad_method == 'none':
            return

        if 'wd' in [body.mesh_method for body in self.bodies]:
            raise NotImplementedError("reflection not yet supported for WD-style meshing")

            # TODO: to support this, each call to libphoebe (both for convex
            # and general case) needs to have an if statement to be replaced by
            # something like the following:

            # fluxes_intrins_and_refl_per_body = libphoebe.mesh_radiosity_problem_triangles_nbody_convex(vertices_per_body,
            #                                                                            triangles_per_body,
            #                                                                            normals_per_body,
            #                                                                            areas_per_body,
            #                                                                            frac_refls_per_body,
            #                                                                            fluxes_intrins_per_body,
            #                                                                            ld_func_and_coeffs,
            #                                                                            self.reflection_method.title()
            #                                                                            )


        # meshes is an object which allows us to easily access and update columns
        # in the meshes *in memory*.  That is meshes.update_columns will propagate
        # back to the current mesh for each body.
        meshes = self.meshes

        # reflection needs bolometric, energy weighted, normal intensities.
        fluxes_intrins_per_body = []
        for starref, body in self.items():
            abs_normal_intensities = passbands.Inorm_bol_bb(Teff=body.mesh.teffs.for_computations,
                                                            atm='blackbody',
                                                            photon_weighted=False)

            fluxes_intrins_per_body.append(abs_normal_intensities * np.pi)

        fluxes_intrins_flat = meshes.pack_column_flat(fluxes_intrins_per_body)

        if np.all([body.is_convex for body in self.bodies]):
            logger.info("handling reflection (convex case), method='{}'".format(self.irrad_method))

            vertices_per_body = meshes.get_column('vertices').values()
            triangles_per_body = meshes.get_column('triangles').values()
            normals_per_body = meshes.get_column('vnormals').values()
            areas_per_body = meshes.get_column('areas').values()
            irrad_frac_refls_per_body = meshes.get_column('irrad_frac_refl', computed_type='for_computations').values()
            teffs_intrins_per_body = meshes.get_column('teffs', computed_type='for_computations').values()

            ld_func_and_coeffs = [tuple([body.ld_func['bol']] + [np.asarray(body.ld_coeffs['bol'])]) for body in self.bodies]

            fluxes_intrins_and_refl_per_body = libphoebe.mesh_radiosity_problem_nbody_convex(vertices_per_body,
                                                                                       triangles_per_body,
                                                                                       normals_per_body,
                                                                                       areas_per_body,
                                                                                       irrad_frac_refls_per_body,
                                                                                       fluxes_intrins_per_body,
                                                                                       ld_func_and_coeffs,
                                                                                       self.irrad_method.title(),
                                                                                       support=b'vertices'
                                                                                       )

            fluxes_intrins_and_refl_flat = meshes.pack_column_flat(fluxes_intrins_and_refl_per_body)

        else:
            logger.info("handling reflection (general case), method='{}'".format(self.irrad_method))

            vertices_flat = meshes.get_column_flat('vertices')
            triangles_flat = meshes.get_column_flat('triangles')
            normals_flat = meshes.get_column_flat('vnormals')
            areas_flat = meshes.get_column_flat('areas')
            irrad_frac_refls_flat = meshes.get_column_flat('irrad_frac_refl', computed_type='for_computations')

            ld_func_and_coeffs = [tuple([body.ld_func['bol']] + [np.asarray(body.ld_coeffs['bol'])]) for body in self.bodies]
            ld_inds_flat = meshes.pack_column_flat({body.comp_no: np.full(fluxes.shape, body.comp_no-1) for body, fluxes in zip(self.bodies, fluxes_intrins_per_body)})

            fluxes_intrins_and_refl_flat = libphoebe.mesh_radiosity_problem(vertices_flat,
                                                                            triangles_flat,
                                                                            normals_flat,
                                                                            areas_flat,
                                                                            irrad_frac_refls_flat,
                                                                            fluxes_intrins_flat,
                                                                            ld_func_and_coeffs,
                                                                            ld_inds_flat,
                                                                            self.irrad_method.title(),
                                                                            support=b'vertices'
                                                                            )



        teffs_intrins_flat = meshes.get_column_flat('teffs', computed_type='for_computations')

        # update the effective temperatures to give this same bolometric
        # flux under stefan-boltzmann. These effective temperatures will
        # then be used for all passband intensities.
        teffs_intrins_and_refl_flat = teffs_intrins_flat * (fluxes_intrins_and_refl_flat / fluxes_intrins_flat) ** (1./4)

        # TODO: set to triangles if WD mesh_method (note: WD mesh will
        # currently raise NotImplementedError)
        meshes.set_column_flat('teffs', teffs_intrins_and_refl_flat)

    def handle_eclipses(self, expose_horizon=True, **kwargs):
        """
        Detect the triangles at the horizon and the eclipsed triangles, handling
        any necessary subdivision.

        :parameter str eclipse_method: name of the algorithm to use to detect
            the horizon or eclipses (defaults to the value set by computeoptions)
        :parameter str subdiv_alg: name of the algorithm to use for subdivision
            (defaults to the value set by computeoptions)
        :parameter int subdiv_num: number of subdivision iterations (defaults
            the value set by computeoptions)
        """

        eclipse_method = kwargs.get('eclipse_method', self.eclipse_method)
        horizon_method = kwargs.get('horizon_method', self.horizon_method)
        # subdiv_alg = kwargs.get('subdiv_alg', self.subdiv_alg)
        # subdiv_num = int(kwargs.get('subdiv_num', self.subdiv_num))

        # Let's first check to see if eclipses are even possible at these
        # positions.  If they are not, then we only have to do horizon
        #
        # To do that, we'll take the conservative max_r for each object
        # and their current positions, and see if the separations are larger
        # than sum of max_rs
        possible_eclipse = False
        if len(self.bodies) == 1:
            if self.bodies[0].__class__.__name__ == 'Envelope':
                possible_eclipse = True
            else:
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

        if not possible_eclipse and not expose_horizon and horizon_method=='boolean':
            eclipse_method = 'only_horizon'

        # meshes is an object which allows us to easily access and update columns
        # in the meshes *in memory*.  That is meshes.update_columns will propogate
        # back to the current mesh for each body.
        meshes = self.meshes

        # Reset all visibilities to be fully visible to start
        meshes.update_columns('visiblities', 1.0)

        ecl_func = getattr(eclipse, eclipse_method)

        if eclipse_method=='native':
            ecl_kwargs = {'horizon_method': horizon_method}
        else:
            ecl_kwargs = {}

        visibilities, weights, horizon = ecl_func(meshes,
                                                  self.xs, self.ys, self.zs,
                                                  expose_horizon=expose_horizon,
                                                  **ecl_kwargs)

        # NOTE: analytic horizons are called in backends.py since they don't
        # actually depend on the mesh at all.

        # visiblilities here is a dictionary with keys being the component
        # labels and values being the np arrays of visibilities.  We can pass
        # this dictionary directly and the columns will be applied respectively.
        meshes.update_columns('visibilities', visibilities)

        if weights is not None:
            meshes.update_columns('weights', weights)

        return horizon


    def observe(self, dataset, kind, components=None, distance=1.0, l3=0.0):
        """
        TODO: add documentation

        Integrate over visible surface elements and return a dictionary of observable values

        distance (m)
        """

        meshes = self.meshes
        if kind=='rv':
            visibilities = meshes.get_column_flat('visibilities', components)

            if np.all(visibilities==0):
                # then no triangles are visible, so we should return nan
                #print "NO VISIBLE TRIANGLES!!!"
                return {'rv': np.nan}

            rvs = meshes.get_column_flat("rvs:{}".format(dataset), components)
            abs_intensities = meshes.get_column_flat('abs_intensities:{}'.format(dataset), components)
            # mus here will be from the tnormals of the triangle and will not
            # be weighted by the visibility of the triangle
            mus = meshes.get_column_flat('mus', components)
            areas = meshes.get_column_flat('areas_si', components)
            ldint = meshes.get_column_flat('ldint:{}'.format(dataset), components)
            # don't need ptfarea because its a float (same for all elements, regardless of component)

            # note that the intensities are already projected but are per unit area
            # so we need to multiply by the /projected/ area of each triangle (thus the extra mu)
            return {'rv': np.average(rvs, weights=abs_intensities*areas*mus*ldint*visibilities)}

        elif kind=='lc':
            visibilities = meshes.get_column_flat('visibilities')

            if np.all(visibilities==0):
                # then no triangles are visible, so we should return nan - probably shouldn't ever happen for lcs
                return {'flux': np.nan}

            intensities = meshes.get_column_flat("intensities:{}".format(dataset), components)
            mus = meshes.get_column_flat('mus', components)
            areas = meshes.get_column_flat('areas_si', components)
            ldint = meshes.get_column_flat('ldint:{}'.format(dataset), components)

            # assume that all bodies are using the same passband and therefore
            # will have the same ptfarea.  If this assumption is ever a problem -
            # then we will need to build a flat column based on the component
            # of each element so that ptfarea is an array with the same shape
            # as those above
            ptfarea = self.bodies[0].get_ptfarea(dataset)  # TODO: what to pass for component for contacts?

            # intens_proj is the intensity in the direction of the observer per unit surface area of the triangle
            # areas is the area of each triangle
            # areas*mus is the area of each triangle projected in the direction of the observer
            # visibilities is 0 for hidden, 0.5 for partial, 1.0 for visible
            # areas*mus*visibilities is the visibile projected area of each triangle (ie half the area for a partially-visible triangle)
            # so, intens_proj*areas*mus*visibilities is the intensity in the direction of the observer per the observed projected area of that triangle
            # and the sum of these values is the observed flux

            # note that the intensities are already projected but are per unit area
            # so we need to multiply by the /projected/ area of each triangle (thus the extra mu)

            return {'flux': np.sum(intensities*areas*mus*visibilities)*ptfarea/(distance**2)+l3}


        elif kind == 'ifm':
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
            raise NotImplementedError("observe for dataset with kind '{}' not implemented".format(kind))




class Body(object):
    def __init__(self, comp_no, ind_self, ind_sibling, masses, ecc,
                 atm='blackbody',
                 datasets=[], passband = {}, intens_weighting='energy',
                 ld_func={}, ld_coeffs={},
                 dynamics_method='keplerian',
                 mesh_init_phi=0.0):
        """
        TODO: add documentation
        """
        self._is_convex = False

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

        # self.mesh will be filled later once a mesh is created and placed in orbit
        self._mesh = None

        # TODO: double check to see if these are still used or can be removed
        self.time = None
        self.populated_at_time = []

        # Let's create a dictionary to store "standard" protomeshes at different "phases"
        # For example, we may want to store the mesh at periastron and use that as a standard
        # for reprojection for volume conservation in eccentric orbits.
        # Storing meshes should only be done through self.save_as_standard_mesh(theta)
        self._standard_meshes = {}

        self.atm = atm

        # DATSET-DEPENDENT DICTS
        self.passband = passband
        self.intens_weighting = intens_weighting
        self.ld_coeffs = ld_coeffs
        self.ld_func = ld_func

        # Let's create a dictionary to handle how each dataset should scale between
        # absolute and relative intensities.
        self._pblum_scale = {}
        self._ptfarea = {}

        # We'll also keep track of a conservative maximum r (from center of star to triangle, in real units).
        # This will be computed and stored when the periastron mesh is added as a standard
        self._max_r = None

        self.mesh_init_phi = mesh_init_phi

        # TODO: allow custom meshes (see alpha:universe.Body.__init__)
        # TODO: reconsider partial/hidden/visible into opacity/visibility

    def copy(self):
        """
        Make a deepcopy of this Mesh object
        """
        return copy.deepcopy(self)

    @property
    def mesh(self):
        """
        TODO: add documentation
        """
        # if not self._mesh:
            # self._mesh = self.get_standard_mesh(scaled=True)

        # NOTE: self.mesh is the SCALED mesh PLACED in orbit at the current
        # time (self.time).  If this isn't available yet, self.mesh will
        # return None (it is reset to None by self.reset_time())
        return self._mesh

    @property
    def is_convex(self):
        """
        :return: whether the mesh can be assumed to be convex
        :rtype: bool
        """
        return self._is_convex


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

        return self.mesh.volume
        # return compute_volume(self.mesh['size'], self.mesh['center'], self.mesh['normal_'])

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
            # then we want the center-of-mass coordinates
            # TODO: clean this up
            return np.average([_value(coords_array[i]) for i in index], weights=[self._get_mass_by_index(i) for i in index])
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
        d = 1 - self.ecc
        new_mesh_dict, scale, mesh_args = self._build_mesh(d=d,
                                                      mesh_method=mesh_method,
                                                      **kwargs)
        self._scale = scale
        self._mesh_args = mesh_args


        N = len(new_mesh_dict['triangles'])

        logger.info("covered surface with %d triangles"%(N))

        protomesh = mesh.ProtoMesh(**new_mesh_dict)

        self.save_as_standard_mesh(protomesh, theta=0.0)

        # TODO NOW: should these be done on the scaled or unscaled protomesh?
        # self._mesh = self.get_standard_mesh(theta=0.0, scaled=True)
        # self._fill_abuns(kwargs.get('abun', self.abun))  # subclassed objects must set self.abun before calling initialize_mesh
        # self._compute_instantaneous_quantities([], [], [], d=d) # TODO: is this Star-specific?
        # self._fill_loggs([], [], [], d=d)
        # self._fill_gravs()
        # self._fill_teffs()

        self._mesh_initialized = True

        mesh_peri = mesh.ScaledProtoMesh.from_proto(protomesh, self._scale)
        # NOTE: this mesh is not placed in orbit, but that is fine since we
        # only need to get the volume (scaled)
        # TODO: use new_mesh['volume'] instead
        self.volume_at_periastron = new_mesh_dict['volume'] #  NOTE: this is the unscaled volume
        # self.volume_at_periastron = mesh_peri.volume

        return

    def save_as_standard_mesh(self, protomesh, theta=0.0):
        """
        TODO: add documentation
        """
        # TODO: change from theta to d?

        self._standard_meshes[theta] = protomesh.copy()

        if theta==0.0:
            # then this is when the object could be most inflated, so let's
            # store the maximum distance to a triangle.  This is then used to
            # conservatively and efficiently estimate whether an eclipse is
            # possible at any given combination of positions
            mesh = self.get_standard_mesh(theta=0.0, scaled=True)

            self._max_r = np.sqrt(max([x**2+y**2+z**2 for x,y,z in mesh.centers]))

    def get_standard_mesh(self, theta=0.0, scaled=True):
        """
        TODO: add documentation
        """
        protomesh = self._standard_meshes[theta] #.copy() # if theta in self._standard_meshes.keys() else self.mesh.copy()

        if scaled:
            return mesh.ScaledProtoMesh.from_proto(protomesh, self._scale)
        else:
            return protomesh.copy()

        # return mesh

    def reset_time(self, time):
        """
        TODO: add documentation
        """
        self._mesh = None
        self.time = time
        self.populated_at_time = []

        return

    def update_position(self, time, xs, ys, zs, vxs, vys, vzs, ethetas, elongans, eincls, ds=None, Fs=None, ignore_effects=False, **kwargs):
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
        :parameter list eincls: a list/array of euler-incls of ALL COMPONENTS in the :class:`System`
        :parameter list ds: (optional) a list/array of instantaneous distances of ALL COMPONENTS in the :class:`System`
        :parameter list Fs: (optional) a list/array of instantaneous syncpars of ALL COMPONENTS in the :class:`System`
        :raises NotImplementedError: if the dynamics_method is not supported
        """
        if not self.mesh_initialized:
            self.initialize_mesh()

        self.reset_time(time)


        #-- Get current position/euler information
        # TODO: remove this if/else if all dynamics method now support eulerian elements
        if self.dynamics_method in ['keplerian', 'nbody', 'rebound', 'bs']:
            # if we can't get the polar direction, assume it's in the negative Z-direction
            try:
                # TODO: implement get_polar_direction (see below for alpha version)
                # this will currently ALWAYS fail and follow the except - meaning
                # misaligned orbits are not yet supported
                polar_dir = -self.get_polar_direction(norm=True)
            except:
                #logger.warning("assuming polar direction - misaligned orbits not yet supported")
                polar_dir = default_polar_dir

            # TODO: get rid of this ugly _value stuff
            pos = (_value(xs[self.ind_self]), _value(ys[self.ind_self]), _value(zs[self.ind_self]))
            vel = (_value(vxs[self.ind_self]), _value(vys[self.ind_self]), _value(vzs[self.ind_self]))
            euler = (_value(ethetas[self.ind_self]), _value(elongans[self.ind_self]), _value(eincls[self.ind_self]))
        else:
            raise NotImplementedError("update_position does not support dynamics_method={}".format(self.dynamics_method))


        #-- Volume Conservation
        if self.needs_volume_conservation and self.distortion_method in ['roche']:

            # TODO: this seems Star/Roche-specific - should it be moved to that class or can it be generalized?

            q, F, d, Phi = self._mesh_args

            # override d to be the current value
            if ds is not None:
                # then the instantaneous sma was likely changing (ie roche geometry but nbody orbits)
                d = ds[self.ind_self]
                # TODO: if we change d here based on the new sma, do we need to update self._scale?
            else:
                d = self.instantaneous_distance(xs, ys, zs, self.sma)

            if Fs is not None:
                # then the instantaneous F was likely changing (ie roche geometry but nbody orbits)
                F = Fs[self.ind_self]

            # TODO: TESTING should this be unscaled with the new scale or old scale?
            # self._scale = d
            target_volume = self.get_target_volume(ethetas[self.ind_self], scaled=False)
            logger.info("volume conservation: target_volume={}".format(target_volume))


            # print "*** libphoebe.roche_Omega_at_vol", target_volume, q, F, d, Phi, self.Phi

            # TODO: need to send a better guess for Omega0
            Phi = libphoebe.roche_Omega_at_vol(target_volume,
                                               q, F, d,
                                               Omega0=Phi if Phi>self.Phi else self.Phi)
            # if Phi < self.Phi:
                # then for some reason we passed the value defined at periastron...
                # NOTE: this logic may not make sense for dynamical system
                # logger.warning("Pot falling back to value defined at periastron")
                # Phi = self.Phi

            # to store this as instantaneous pot, we need to translate back to the secondary ref frame if necessary
            self._instantaneous_pot = roche.pot_for_component(Phi, self.q, self.comp_no)

            #-- Reprojection
            logger.info("reprojecting mesh onto Phi={} at d={}".format(Phi, d))

            # TODO: implement reprojection as an option instead of rebuilding
            # the mesh??

            # TODO: make sure this passes the new d and new Phi correctly

            # NOTE: Phi is not Phi_user so doesn't need to be flipped for the
            # secondary component
            new_mesh_dict, scale, mesh_args = self._build_mesh(d=d,
                                                               F=F,
                                                               mesh_method=self.mesh_method,
                                                               Phi=Phi)
            # TODO: do we need to update self.scale or self._mesh_args???
            # TODO: need to be very careful about self.sma vs self._scale - maybe need to make a self._instantaneous_scale???
            self._scale = scale
            self._mesh_args = mesh_args


            # Here we'll build a scaledprotomesh directly from the newly
            # marched mesh since we don't need to store the protomesh itself
            # as a new standard.  NOTE that we're using scale from the new
            # mesh rather than self._scale since the instantaneous separation
            # has likely changed since periastron
            scaledprotomesh = mesh.ScaledProtoMesh(scale=scale, **new_mesh_dict)


        else:

            # We still need to go through scaledprotomesh instead of directly
            # to mesh since features may want to process the body-centric
            # coordinates before placing in orbit
            scaledprotomesh = self.get_standard_mesh(scaled=True)
            # TODO: can we avoid an extra copy here?


        if not ignore_effects:
            # First allow features to edit the coords_for_computations (pvertices).
            # Changes here WILL affect future computations for logg, teff,
            # intensities, etc.  Note that these WILL NOT affect the
            # coords_for_observations automatically - those should probably be
            # perturbed as well, unless there is a good reason not to.
            for feature in self.features:
                # NOTE: these are ALWAYS done on the protomesh
                coords_for_observations = feature.process_coords_for_computations(scaledprotomesh.coords_for_computations, t=self.time)
                if scaledprotomesh._compute_at_vertices:
                    scaledprotomesh.update_columns(pvertices=coords_for_observations)

                else:
                    scaledprotomesh.update_columns(centers=coords_for_observations)
                    raise NotImplementedError("areas are not updated for changed mesh")


            for feature in self.features:
                coords_for_observations = feature.process_coords_for_observations(scaledprotomesh.coords_for_computations, scaledprotomesh.coords_for_observations, t=self.time)
                if scaledprotomesh._compute_at_vertices:
                    scaledprotomesh.update_columns(vertices=coords_for_observations)

                    # TODO: centers either need to be supported or we need to report
                    # vertices in the frontend as x, y, z instead of centers

                    updated_props = libphoebe.mesh_properties(scaledprotomesh.vertices,
                                                              scaledprotomesh.triangles,
                                                              tnormals=True,
                                                              areas=True)

                    scaledprotomesh.update_columns(**updated_props)

                else:
                    scaledprotomesh.update_columns(centers=coords_for_observations)
                    raise NotImplementedError("areas are not updated for changed mesh")


        # print "*** scaledprotomesh.vz", scaledprotomesh.velocities.for_computations[:,2].min(), scaledprotomesh.velocities.for_computations[:,2].max(), scaledprotomesh.velocities.for_computations[:,2].mean()
        # TODO NOW [OPTIMIZE]: get rid of the deepcopy here - but without it the mesh velocities build-up and do terrible things
        self._mesh = mesh.Mesh.from_scaledproto(scaledprotomesh.copy(),
                                                pos, vel, euler,
                                                polar_dir*self.freq_rot)

        # print "time: {}\npos: {}\nvel: {}\neuler: {}".format(time, pos, vel, euler)
        # print "*** mesh.vz", self._mesh.velocities.for_computations[:,2].min(), self._mesh.velocities.for_computations[:,2].max(), self._mesh.velocities.for_computations[:,2].mean()




        # Lastly, we'll recompute physical quantities (not observables) if
        # needed for this time-step.
        # TODO: make sure features smartly trigger needs_recompute_instantaneous
        if self.mesh.loggs.for_computations is None or self.needs_recompute_instantaneous:
            self._compute_instantaneous_quantities(xs, ys, zs)

            # Now fill local instantaneous quantities
            self._fill_loggs(ignore_effects=ignore_effects)
            self._fill_gravs()
            self._fill_teffs(ignore_effects=ignore_effects)
            self._fill_abuns(abun=self.abun)
            self._fill_albedos(irrad_frac_refl=self.irrad_frac_refl)


        return

    def _fill_abuns(self, mesh=None, abun=0.0):
        """
        TODO: add documentation
        """
        if mesh is None:
            mesh = self.mesh

        # TODO: support from frontend

        mesh.update_columns(abuns=abun)

    def _fill_albedos(self, mesh=None, irrad_frac_refl=0.0):
        """
        TODO: add documentation
        """
        if mesh is None:
            mesh = self.mesh

        mesh.update_columns(irrad_frac_refl=irrad_frac_refl)


    def compute_luminosity(self, dataset, **kwargs):
        """
        """
        # areas are the NON-projected areas of each surface element.  We'll be
        # integrating over normal intensities, so we don't need to worry about
        # multiplying by mu to get projected areas.
        areas = self.mesh.areas_si

        # abs_normal_intensities are directly out of the passbands module and are
        # emergent normal intensities in this dataset's passband/atm in absolute units
        abs_normal_intensities = self.mesh['abs_normal_intensities:{}'.format(dataset)].centers

        ldint = self.mesh['ldint:{}'.format(dataset)].centers
        ptfarea = self.get_ptfarea(dataset) # just a float

        # Our total integrated intensity in absolute units (luminosity) is now
        # simply the sum of the normal emergent intensities times pi (to account
        # for intensities emitted in all directions across the solid angle),
        # limbdarkened as if they were at mu=1, and multiplied by their respective areas

        abs_luminosity = np.sum(abs_normal_intensities*areas*ldint)*ptfarea*np.pi

        # NOTE: when this is computed the first time (for the sake of determining
        # pblum_scale), get_pblum_scale will return 1.0
        return abs_luminosity * self.get_pblum_scale(dataset)

    def compute_pblum_scale(self, dataset, pblum, **kwargs):
        """
        intensities should already be computed for this dataset at the time for which pblum is being provided

        TODO: add documentation
        """

        abs_luminosity = self.compute_luminosity(dataset, **kwargs)

        # We now want to remember the scale for all intensities such that the
        # luminosity in relative units gives the provided pblum
        pblum_scale = pblum / abs_luminosity

        self.set_pblum_scale(dataset, pblum_scale)

    def set_pblum_scale(self, dataset, pblum_scale, **kwargs):
        self._pblum_scale[dataset] = pblum_scale

    def get_pblum_scale(self, dataset, **kwargs):
        """
        """
        # kwargs needed just so component can be passed but ignored

        if dataset in self._pblum_scale.keys():
            return self._pblum_scale[dataset]
        else:
            #logger.warning("no pblum scale found for dataset: {}".format(dataset))
            return 1.0

    def set_ptfarea(self, dataset, ptfarea, **kwargs):
        """
        """
        self._ptfarea[dataset] = ptfarea

    def get_ptfarea(self, dataset, **kwargs):
        """
        """
        # kwargs needed just so component can be passed but ignored

        return self._ptfarea[dataset]


    def populate_observable(self, time, kind, dataset, **kwargs):
        """
        TODO: add documentation
        """

        if kind in ['mesh']:
            return

        if time==self.time and dataset in self.populated_at_time and 'pblum' not in kind:
            # then we've already computed the needed columns

            # TODO: handle the case of intensities already computed by /different/ dataset (ie RVs computed first and filling intensities and then lc requesting intensities with SAME passband/atm)
            return

        new_mesh_cols = getattr(self, '_populate_{}'.format(kind.lower()))(dataset, **kwargs)

        for key, col in new_mesh_cols.items():

            self.mesh.update_columns_dict({'{}:{}'.format(key, dataset): col})

        self.populated_at_time.append(dataset)

    # def _populate_lc_pblum(self, dataset, **kwargs):

    #     abs_normal_intensities = self.mesh.observables['abs_normal_intensities:{}'.format(dataset)].for_computations
    #     abs_intensities = self.mesh.observables['abs_intensities:{}'.format(dataset)].for_computations


    #     intensities = abs_intensities * self.get_pblum_scale(dataset)

    #     return {'normal_intensities': normal_intensities,
    #             'intensities': intensities}

class CustomBody(Body):
    def __init__(self, masses, sma, ecc, freq_rot, teff, abun,
                 dynamics_method='keplerian',
                 ind_self=0, ind_sibling=1, comp_no=1,
                 atm='blackbody', datasets=[], passband={},
                 intens_weighting={}, ld_func={}, ld_coeffs={}, **kwargs):
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
        super(CustomBody, self).__init__(comp_no, ind_self, ind_sibling,
                                         masses, ecc,
                                         atm, datasets, passband,
                                         intens_weighting,
                                         ld_func, ld_coeffs,
                                         dynamics_method=dynamics_method)


        self.teff = teff
        self.abun = abun


        self.sma = sma


        raise NotImplementedError


    @classmethod
    def from_bundle(cls, b, component, compute=None,
                    dynamics_method='keplerian', datasets=[], **kwargs):
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

        # TODO: retrieve atm, ld_func, ld_coeffs
        atm = 'blackbody'
        ld_func = {}
        ld_coeffs = {}
        passband = {}
        intens_weighting = {}

        return cls(masses, sma, ecc, freq_rot, teff, abun, dynamics_method,
                   ind_self, ind_sibling, comp_no,
                   atm, datasets, passband, intens_weighting, ld_func, ld_coeffs)


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

        return new_mesh, sma, mesh_args

    def _fill_teffs(self, ignore_effects=False, **kwargs):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """

        self.mesh.update_columns(teffs=self.teff)

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

        self.set_ptfarea(dataset, ptfarea)

        return {'abs_normal_intensities': abs_normal_intensities,
                'normal_intensities': normal_intensities,
                'abs_intensities': abs_intensities,
                'intensities': intensities,
                'ldint': ldint}


class Star(Body):
    def __init__(self, F, Phi, masses, sma, ecc, freq_rot, teff, gravb_bol,
                 abun, irrad_frac_refl,
                 mesh_method='marching',
                 dynamics_method='keplerian',
                 mesh_init_phi=0.0,
                 ind_self=0, ind_sibling=1,
                 comp_no=1, is_single=False,
                 atm='blackbody', datasets=[], passband={},
                 intens_weighting={}, ld_func={}, ld_coeffs={},
                 do_rv_grav=False,
                 features=[], do_mesh_offset=True, **kwargs):
        """

        :parameter float F: syncpar
        :parameter float Phi: equipotential of this star at periastron
        :parameter masses: mass of each component in the system (solMass)
        :type masses: list of floats
        :parameter float sma: sma of this component's parent orbit (solRad)
        :parameter float freq_rot: rotation frequency (rad/d)
        :parameter float abun: abundance of this star
        :parameter int ind_self: index in all arrays (positions, masses, etc) for this object
        :parameter int ind_sibling: index in all arrays (positions, masses, etc)
            for the sibling of this object
        :return: instantiated :class:`Star` object
        """
        super(Star, self).__init__(comp_no, ind_self, ind_sibling, masses, ecc,
                                   atm, datasets, passband,
                                   intens_weighting, ld_func, ld_coeffs,
                                   dynamics_method=dynamics_method,
                                   mesh_init_phi=mesh_init_phi)

        self._is_convex = True

        # Remember how to compute the mesh
        self.mesh_method = mesh_method
        self.ntriangles = kwargs.get('ntriangles', 1000)                    # Marching
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
        # NOTE: self.q may be flipped her for the secondary
        self.Phi = roche.pot_for_component(Phi, self.q, self.comp_no)

        self.teff = teff
        self.gravb_bol = gravb_bol
        # self.gravb_law = gravb_law
        self.abun = abun
        self.irrad_frac_refl = irrad_frac_refl

        # self.frac_heat = frac_heat
        # self.frac_scatt = frac_scatt

        self.features = features

        self._is_single = is_single # TODO: move to Body class?
        self._do_mesh_offset = do_mesh_offset

        # Volume "conservation"
        self.volume_factor = 1.0  # TODO: eventually make this a parameter (currently defined to be the ratio between volumes at apastron/periastron)


    @classmethod
    def from_bundle(cls, b, component, compute=None, dynamics_method='keplerian',
                    mesh_init_phi=0.0, datasets=[], **kwargs):
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

        self_ps = b.filter(component=component, context='component', check_visible=False)
        F = self_ps.get_value('syncpar', check_visible=False) # not relevant for single stars... but doesn't hurt to load
        Phi = self_ps.get_value('pot')
        freq_rot = self_ps.get_value('freq', unit=u.rad/u.d)
        # NOTE: we need F for roche geometry (marching, reprojection), but freq_rot for ctrans.place_in_orbit and rotstar.marching


        masses = [b.get_value('mass', component=star, context='component', unit=u.solMass) for star in starrefs]
        if b.hierarchy.get_parent_of(component) != 'component':
            sma = b.get_value('sma', component=label_orbit, context='component', unit=u.solRad)
            ecc = b.get_value('ecc', component=label_orbit, context='component')
            is_single = False
        else:
            # single star case
            # here sma is meaningless, but we'll compute the mesh using the polar radius as the scaling factor
            sma = 1.0 #b.get_value('rpole', component=star, context='component', unit=u.solRad)
            ecc = 0.0
            is_single = True

        teff = b.get_value('teff', component=component, context='component', unit=u.K)
        # gravb_law = b.get_value('gravblaw_bol', component=component, context='component')
        gravb_bol= b.get_value('gravb_bol', component=component, context='component')

        abun = b.get_value('abun', component=component, context='component')
        irrad_frac_refl = b.get_value('irrad_frac_refl_bol', component=component, context='component')

        try:
            do_rv_grav = b.get_value('rv_grav', component=component, compute=compute, check_visible=False, **kwargs) if compute is not None else False
        except ValueError:
            # rv_grav may not have been copied to this component if no rvs are attached
            do_rv_grav = False

        # pass kwargs in case mesh_method was temporarily overriden
        mesh_method = b.get_value('mesh_method', component=component, compute=compute, **kwargs) if compute is not None else 'marching'

        mesh_kwargs = {}
        if mesh_method == 'marching':
            mesh_kwargs['ntriangles'] = b.get_value('ntriangles', component=component, compute=compute, **kwargs) if compute is not None else 1000
            mesh_kwargs['distortion_method'] = b.get_value('distortion_method', component=component, compute=compute, **kwargs) if compute is not None else 'roche'
        elif mesh_method == 'wd':
            mesh_kwargs['gridsize'] = b.get_value('gridsize', component=component, compute=compute, **kwargs) if compute is not None else 30
        else:
            raise NotImplementedError

        features = []
        # print "*** checking for features of", component, b.filter(component=component).features
        for feature in b.filter(component=component).features:
            # print "*** creating features", star, feature
            feature_ps = b.filter(feature=feature, component=component)
            feature_cls = globals()[feature_ps.kind.title()]
            features.append(feature_cls.from_bundle(b, feature))

        if conf.devel:
            do_mesh_offset = b.get_value('mesh_offset', compute=compute, **kwargs)
        else:
            do_mesh_offset = True

        datasets_intens = [ds for ds in b.filter(kind=['lc', 'rv', 'ifm'], context='dataset').datasets if ds != '_default']
        atm = b.get_value('atm', compute=compute, component=component, **kwargs) if compute is not None else 'blackbody'
        passband = {ds: b.get_value('passband', dataset=ds, **kwargs) for ds in datasets_intens}
        intens_weighting = {ds: b.get_value('intens_weighting', dataset=ds, **kwargs) for ds in datasets_intens}
        ld_func = {ds: b.get_value('ld_func', dataset=ds, component=component, **kwargs) for ds in datasets_intens}
        ld_coeffs = {ds: b.get_value('ld_coeffs', dataset=ds, component=component, check_visible=False, **kwargs) for ds in datasets_intens}
        ld_func['bol'] = b.get_value('ld_func_bol', component=component, context='component', **kwargs)
        ld_coeffs['bol'] = b.get_value('ld_coeffs_bol', component=component, context='component', **kwargs)

        return cls(F, Phi, masses, sma, ecc, freq_rot, teff, gravb_bol,
                abun, irrad_frac_refl,
                mesh_method, dynamics_method,
                mesh_init_phi, ind_self, ind_sibling, comp_no,
                is_single=is_single, atm=atm, datasets=datasets,
                passband=passband, intens_weighting=intens_weighting,
                ld_func=ld_func, ld_coeffs=ld_coeffs,
                do_rv_grav=do_rv_grav,
                features=features, do_mesh_offset=do_mesh_offset, **mesh_kwargs)

    @property
    def spots(self):
        return [f for f in self.features if f.__class__.__name__=='Spot']

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
        even for circular orbits - if we're using nbody but roche distortion, we must remesh to handle instantaneous changes to d or F
        """
        return self.ecc != 0 or (self.dynamics_method != 'keplerian' and self.distortion_method == 'roche')


    def get_target_volume(self, etheta, scaled=False):
        """
        TODO: add documentation

        get the volume that the BRS should have at a given euler theta
        """
        # TODO: make this a function of d instead of etheta?
        logger.info("determining target volume at theta={}".format(etheta))

        # TODO: eventually this could allow us to "break" volume conservation and have volume be a function of d,
        # with some scaling factor provided by the user as a parameter.  Until then, we'll assume volume is conserved
        # which means the volume should always be the same as it was defined at periaston.

        # volumes are stored internally in real units.  So if we want the
        # "protomesh" volume we need to divide by scale^3
        if not scaled:
            # return self.volume_at_periastron/self._scale**3
            return self.volume_at_periastron
        else:
            # return self.volume_at_periastron
            return self.volume_at_periastron*self._scale**3

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

        mesh_args = (q, F, d, Phi)

        if mesh_method == 'marching':
            ntriangles = kwargs.get('ntriangles', self.ntriangles)

            if self.distortion_method == 'roche':
                # TODO: check whether roche or misaligned roche from values of incl, etc!!!!

                av = libphoebe.roche_area_volume(*mesh_args,
                                                 choice=0,
                                                 larea=True,
                                                 lvolume=True)
                delta = np.sqrt(4./np.sqrt(3) * av['larea'] / ntriangles)

                # print "*** libphoebe.roche_marching_mesh args: {}, rpole: {}, delta: {}".format(mesh_args, rpole, delta)

                new_mesh = libphoebe.roche_marching_mesh(*mesh_args,
                                                         delta=delta,
                                                         choice=0,
                                                         full=True,
                                                         max_triangles=ntriangles*2,
                                                         vertices=True,
                                                         triangles=True,
                                                         centers=True,
                                                         vnormals=True,
                                                         tnormals=True,
                                                         cnormals=False,
                                                         vnormgrads=True,
                                                         cnormgrads=False,
                                                         areas=True,
                                                         volume=False,
                                                         init_phi=self.mesh_init_phi)


                # Now we'll get the area and volume of the Roche potential
                # itself (not the mesh).
                # TODO: which volume(s) do we want to report?  Either way, make
                # sure to do the same for the OC case and rotstar


                new_mesh['volume'] = av['lvolume']

                if self._do_mesh_offset:
                    # vertices directly from meshing are placed directly on the
                    # potential, causing the volume and surface area to always
                    # (for convex surfaces) be underestimated.  Now let's jitter
                    # each of the vertices along their normals to recover the
                    # expected volume/surface area.  Since they are moved along
                    # their normals, vnormals applies to both vertices and
                    # pvertices.
                    new_mesh['pvertices'] = new_mesh.pop('vertices')
                    # TODO: fall back on curvature=False if we know the body
                    # is relatively spherical
                    mo = libphoebe.mesh_offseting(av['larea'],
                                                  new_mesh['pvertices'],
                                                  new_mesh['vnormals'],
                                                  new_mesh['triangles'],
                                                  curvature=True,
                                                  vertices=True,
                                                  tnormals=True,
                                                  areas=True,
                                                  volume=False)

                    new_mesh['vertices'] = mo['vertices']
                    new_mesh['areas'] = mo['areas']
                    new_mesh['tnormals'] = mo['tnormals']

                    # TODO: need to update centers (so that they get passed
                    # to the frontend as x, y, z)
                    # new_mesh['centers'] = mo['centers']


                else:
                    # pvertices should just be a copy of vertice
                    new_mesh['pvertices'] = new_mesh['vertices']



                # We only need the gradients where we'll compute local
                # quantities which, for a marching mesh, is at the vertices.
                new_mesh['normgrads'] = new_mesh.pop('vnormgrads')

                # And lastly, let's fill the velocities column - with zeros
                # at each of the vertices
                new_mesh['velocities'] = np.zeros(new_mesh['vertices'].shape)

                new_mesh['tareas'] = np.array([])

                scale = sma


            elif self.distortion_method == 'rotstar':

                if not self._is_single:
                    # Then we're in a binary that is using the roche
                    # pot<->rpole constraint.  So, let's get the Phi that
                    # would match the same polar radius as if it were a roche
                    # potential.
                    # rpole = roche.potential2rpole(Phi, q, 0.0, F)  # TODO: REMOVE
                    # print "*** before rotstar_from_roche", Phi, F, sma, rpole*sma
                    omega, Phi = libphoebe.rotstar_from_roche(*mesh_args)
                    # rpole = rotstar.potential2rpole(Phi, self.freq_rot, solar_units=True)  # TODO: REMOVE
                    # print "*** after rotstar_from_roche", Phi, omega, sma, rpole*sma

                else:
                    # then we used the rotstar pot<->rpole constraint and
                    # can directly pass Phi and omega (from freq_rot)

                    # freq_rot (1./d)
                    omega = rotstar.rotfreq_to_omega(self.freq_rot, scale=sma, solar_units=True)
                    Phi = self.Phi_user # because we don't want to do conversion for secondary


                mesh_args = (omega, Phi)

                av = libphoebe.rotstar_area_volume(*mesh_args,
                                                   larea=True,
                                                   lvolume=True)

                delta = np.sqrt(4./np.sqrt(3) * av['larea'] / ntriangles)


                # print "*** rotstar_marching_mesh omega: {}, Phi: {}, freq_rot:{}, sma:{}, rpole:{}, delta:{}".format(mesh_args[0], mesh_args[1], self.freq_rot, sma, rpole, delta)

                new_mesh = libphoebe.rotstar_marching_mesh(*mesh_args,
                                               delta=delta,
                                               full=True,
                                               max_triangles=ntriangles*2,
                                               vertices=True,
                                               triangles=True,
                                               centers=True,
                                               vnormals=True,
                                               tnormals=True,
                                               cnormals=False,
                                               vnormgrads=True,
                                               cnormgrads=False,
                                               areas=True,
                                               volume=True,
                                               init_phi=self.mesh_init_phi)



                new_mesh['volume'] = av['lvolume']

                if self._do_mesh_offset:
                    # vertices directly from meshing are placed directly on the
                    # potential, causing the volume and surface area to always
                    # (for convex surfaces) be underestimated.  Now let's jitter
                    # each of the vertices along their normals to recover the
                    # expected volume/surface area.  Since they are moved along
                    # their normals, vnormals applies to both vertices and
                    # pvertices.
                    new_mesh['pvertices'] = new_mesh.pop('vertices')
                    mo = libphoebe.mesh_offseting(av['larea'],
                                                  new_mesh['pvertices'],
                                                  new_mesh['vnormals'],
                                                  new_mesh['triangles'],
                                                  curvature=True,
                                                  vertices=True,
                                                  tnormals=False,
                                                  areas=True,
                                                  volume=False)

                    new_mesh['vertices'] = mo['vertices']
                    new_mesh['areas'] = mo['areas']

                    # TODO: need to update centers (so that they get passed
                    # to the frontend as x, y, z)
                    # new_mesh['centers'] = mo['centers']

                else:
                    # pvertices should just be a copy of vertices
                    new_mesh['pvertices'] = new_mesh['vertices']

                # We only need the gradients where we'll compute local
                # quantities which, for a marching mesh, is at the vertices.
                new_mesh['normgrads'] = new_mesh.pop('vnormgrads')

                # And lastly, let's fill the velocities column - with zeros
                # at each of the vertices
                new_mesh['velocities'] = np.zeros(new_mesh['vertices'].shape)

                new_mesh['tareas'] = np.array([])

                scale = sma

            elif self.distortion_method == 'sphere':

                if self._is_single:
                    raise NotImplementedError()
                    # TODO: need to use rpole directly?

                rpole = libphoebe.roche_pole(*mesh_args)
                omega = 1./rpole
                mesh_args = (omega,)

                av = libphoebe.sphere_area_volume(*mesh_args,
                                                   larea=True,
                                                   lvolume=True)

                delta = np.sqrt(4./np.sqrt(3) * av['larea'] / ntriangles)

                # print "*** sphere_marching_mesh rpole:{} omega:{} delta:{}".format(rpole, mesh_args[0], delta)

                new_mesh = libphoebe.sphere_marching_mesh(*mesh_args,
                                               delta=delta,
                                               full=True,
                                               max_triangles=ntriangles*2,
                                               vertices=True,
                                               triangles=True,
                                               centers=True,
                                               vnormals=True,
                                               tnormals=True,
                                               cnormals=False,
                                               vnormgrads=True,
                                               cnormgrads=False,
                                               areas=True,
                                               volume=True,
                                               init_phi=self.mesh_init_phi)



                new_mesh['volume'] = av['lvolume']

                if self._do_mesh_offset:
                    # vertices directly from meshing are placed directly on the
                    # potential, causing the volume and surface area to always
                    # (for convex surfaces) be underestimated.  Now let's jitter
                    # each of the vertices along their normals to recover the
                    # expected volume/surface area.  Since they are moved along
                    # their normals, vnormals applies to both vertices and
                    # pvertices.
                    new_mesh['pvertices'] = new_mesh.pop('vertices')
                    mo = libphoebe.mesh_offseting(av['larea'],
                                                  new_mesh['pvertices'],
                                                  new_mesh['vnormals'],
                                                  new_mesh['triangles'],
                                                  curvature=True,
                                                  vertices=True,
                                                  tnormals=False,
                                                  areas=True,
                                                  volume=False)

                    new_mesh['vertices'] = mo['vertices']
                    new_mesh['areas'] = mo['areas']

                    # TODO: need to update centers (so that they get passed
                    # to the frontend as x, y, z)
                    # new_mesh['centers'] = mo['centers']

                else:
                    # pvertices should just be a copy of vertices
                    new_mesh['pvertices'] = new_mesh['vertices']

                # We only need the gradients where we'll compute local
                # quantities which, for a marching mesh, is at the vertices.
                new_mesh['normgrads'] = new_mesh.pop('vnormgrads')

                # And lastly, let's fill the velocities column - with zeros
                # at each of the vertices
                new_mesh['velocities'] = np.zeros(new_mesh['vertices'].shape)

                new_mesh['tareas'] = np.array([])

                scale = sma


            elif self.distortion_method == 'nbody':
                # TODO: implement this (discretize and save mesh_args)
                raise NotImplementedError("'nbody' distortion_method not yet supported - try roche")
            else:
                raise NotImplementedError

        elif mesh_method == 'wd':

            N = int(kwargs.get('gridsize', self.gridsize))

            the_grid = potentials.discretize_wd_style(N, *mesh_args)
            new_mesh = mesh.wd_grid_to_mesh_dict(the_grid, q, F, d)
            scale = sma

        else:
            raise NotImplementedError("mesh_method '{}' is not supported".format(mesh_method))

        return new_mesh, scale, mesh_args

    def _compute_instantaneous_quantities(self, xs, ys, zs, **kwargs):
        """
        TODO: add documentation
        """
        pole_func = getattr(libphoebe, '{}_pole'.format(self.distortion_method))
        gradOmega_func = getattr(libphoebe, '{}_gradOmega_only'.format(self.distortion_method))

        r_pole = pole_func(*self._mesh_args)
        r_pole_ = np.array([0., 0., r_pole])
        args = list(self._mesh_args)[:-1]+[r_pole_]
        grads = gradOmega_func(*args)
        g_pole = np.linalg.norm(grads)

        g_rel_to_abs = c.G.si.value*c.M_sun.si.value*self.masses[self.ind_self]/(self.sma*c.R_sun.si.value)**2*100. # 100 for m/s**2 -> cm/s**2

        self._instantaneous_gpole = g_pole * g_rel_to_abs
        # TODO NOW: check whether r_pole is in absolute units (scaled/not scaled)
        self._instantaneous_rpole = r_pole

    def _fill_loggs(self, mesh=None, ignore_effects=False):
        """
        TODO: add documentation

        Calculate local surface gravity

        GMSunNom = 1.3271244e20 m**3 s**-2
        RSunNom = 6.597e8 m
        """
        if mesh is None:
            mesh = self.mesh

        g_rel_to_abs = c.G.si.value*c.M_sun.si.value*self.masses[self.ind_self]/(self.sma*c.R_sun.si.value)**2*100. # 100 for m/s**2 -> cm/s**2

        loggs = np.log10(mesh.normgrads.for_computations * g_rel_to_abs)

        if not ignore_effects:
            for feature in self.features:
                if feature.proto_coords:
                    loggs = feature.process_teffs(loggs, self.get_standard_mesh().coords_for_computations, t=self.time)
                else:
                    loggs = feature.process_teffs(loggs, mesh.coords_for_computations, t=self.time)

        mesh.update_columns(loggs=loggs)

        # logger.info("derived surface gravity: %.3f <= log g<= %.3f (g_p=%s and Rp=%s Rsol)"%(loggs.min(), loggs.max(), self._instantaneous_gpole, self._instantaneous_rpole*self._scale))

    def _fill_gravs(self, mesh=None, **kwargs):
        """
        TODO: add documentation

        requires _fill_loggs to have been called
        """
        if mesh is None:
            mesh = self.mesh


        # TODO: rename 'gravs' to 'gdcs' (gravity darkening corrections)

        g_rel_to_abs = c.G.si.value*c.M_sun.si.value*self.masses[self.ind_self]/(self.sma*c.R_sun.si.value)**2*100. # 100 for m/s**2 -> cm/s**2
        # TODO: check the division by 100 - is this just to change units back to m?
        gravs = ((mesh.normgrads.for_computations * g_rel_to_abs)/self._instantaneous_gpole)**self.gravb_bol

        # TODO: make sure equivalent to the old way here
        # gravs = abs(10**(self.mesh.loggs.for_computations-2)/self._instantaneous_gpole)**self.gravb_bol

        mesh.update_columns(gravs=gravs)


    def _fill_teffs(self, mesh=None, ignore_effects=False, **kwargs):
        r"""

        requires _fill_loggs and _fill_gravs to have been called

        Calculate local temperature of a Star.
        """
        if mesh is None:
            mesh = self.mesh

        # get the user-defined mean effective temperatures
        Teff = kwargs.get('teff', self.teff)

        # Convert from mean to polar by dividing total flux by gravity darkened flux (Ls drop out)
        # see PHOEBE Legacy scientific reference eq 5.20
        Tpole = Teff*(np.sum(mesh.areas) / np.sum(mesh.gravs.centers*mesh.areas))**(0.25)
        self._instantaneous_teffpole = Tpole

        # Now we can compute the local temperatures.
        # see PHOEBE Legacy scientific reference eq 5.23
        teffs = Tpole*mesh.gravs.for_computations**0.25

        if not ignore_effects:
            for feature in self.features:
                if feature.proto_coords:
                    teffs = feature.process_teffs(teffs, self.get_standard_mesh().coords_for_computations, t=self.time)
                else:
                    teffs = feature.process_teffs(teffs, mesh.coords_for_computations, t=self.time)

        mesh.update_columns(teffs=teffs)

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
        passband = kwargs.get('passband', self.passband[dataset])
        intens_weighting = kwargs.get('intens_weighting', self.intens_weighting.get(dataset, None))
        ld_func = kwargs.get('ld_func', self.ld_func[dataset])
        ld_coeffs = kwargs.get('ld_coeffs', self.ld_coeffs[dataset]) if ld_func != 'interp' else None
        atm = kwargs.get('atm', self.atm)

        lc_cols = self._populate_lc(dataset, **kwargs)

        # lc cols should do for interferometry - at least for continuum
        cols = lc_cols

        return cols

    def _populate_rv(self, dataset, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`
        """

        # We need to fill all the flux-related columns so that we can weigh each
        # triangle's rv by its flux in the requested passband.
        lc_cols = self._populate_lc(dataset, **kwargs)

        # rv per element is just the z-component of the velocity vectory.  Note
        # the change in sign from our right-handed system to rv conventions.
        # These will be weighted by the fluxes when integrating

        rvs = -1*self.mesh.velocities.for_computations[:,2]


        # Gravitational redshift
        if self.do_rv_grav:
            rv_grav = c.G*(self.mass*u.solMass)/(self._instantaneous_rpole*u.solRad)/c.c
            # rvs are in solrad/d internally
            rv_grav = rv_grav.to('solRad/d').value

            rvs += rv_grav

        cols = lc_cols
        cols['rvs'] = rvs
        return cols


    def _populate_lc(self, dataset, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`

        :raises NotImplementedError: if lc_method is not supported
        """

        lc_method = kwargs.get('lc_method', 'numerical')  # TODO: make sure this is actually passed

        passband = kwargs.get('passband', self.passband.get(dataset, None))
        intens_weighting = kwargs.get('intens_weighting', self.intens_weighting.get(dataset, None))
        ld_func = kwargs.get('ld_func', self.ld_func.get(dataset, None))
        ld_coeffs = kwargs.get('ld_coeffs', self.ld_coeffs.get(dataset, None)) if ld_func != 'interp' else None
        atm = kwargs.get('atm', self.atm)
        boosting_method = kwargs.get('boosting_method', self.boosting_method)

        pblum = kwargs.get('pblum', 4*np.pi)

        if lc_method=='numerical':

            pb = passbands.get_passband(passband)

            if intens_weighting=='photon':
                ptfarea = pb.ptf_photon_area/pb.h/pb.c
            else:
                ptfarea = pb.ptf_area

            self.set_ptfarea(dataset, ptfarea)

            ldint = pb.ldint(Teff=self.mesh.teffs.for_computations,
                             logg=self.mesh.loggs.for_computations,
                             abun=self.mesh.abuns.for_computations,
                             atm=atm,
                             ld_func=ld_func,
                             ld_coeffs=ld_coeffs,
                             photon_weighted=intens_weighting=='photon')


            # abs_normal_intensities are the normal emergent passband intensities:
            abs_normal_intensities = pb.Inorm(Teff=self.mesh.teffs.for_computations,
                                              logg=self.mesh.loggs.for_computations,
                                              abun=self.mesh.abuns.for_computations,
                                              atm=atm,
                                              ldint=ldint,
                                              photon_weighted=intens_weighting=='photon')

            # abs_intensities are the projected (limb-darkened) passband intensities
            # TODO: why do we need to use abs(mus) here?
            abs_intensities = pb.Imu(Teff=self.mesh.teffs.for_computations,
                                     logg=self.mesh.loggs.for_computations,
                                     abun=self.mesh.abuns.for_computations,
                                     mu=abs(self.mesh.mus_for_computations),
                                     atm=atm,
                                     ldint=ldint,
                                     ld_func=ld_func,
                                     ld_coeffs=ld_coeffs,
                                     photon_weighted=intens_weighting=='photon')


            # Beaming/boosting
            if boosting_method == 'none':
                boost_factors = 1.0
            elif boosting_method == 'linear':
                bindex = pb.bindex(Teff=self.mesh.teffs.for_computations,
                                   logg=self.mesh.loggs.for_computations,
                                   abun=self.mesh.abuns.for_computations,
                                   mu=abs(self.mesh.mus_for_computations),
                                   atm=atm,
                                   photon_weighted=intens_weighting=='photon')

                boost_factors = 1.0 + bindex * self.mesh.velocities.for_computations[:,2]/37241.94167601236
            else:
                raise NotImplementedError("boosting_method='{}' not supported".format(self.boosting_method))

            # boosting is aspect dependent so we don't need to correct the
            # normal intensities
            abs_intensities *= boost_factors

            # Handle pblum - distance and l3 scaling happens when integrating (in observe)
            # we need to scale each triangle so that the summed normal_intensities over the
            # entire star is equivalent to pblum / 4pi
            normal_intensities = abs_normal_intensities * self.get_pblum_scale(dataset)
            intensities = abs_intensities * self.get_pblum_scale(dataset)



        elif lc_method=='analytical':
            raise NotImplementedError("analytical fluxes not yet ported to beta")
            #lcdep, ref = system.get_parset(ref)
            # The projected intensity is normalised with the distance in cm, we need
            # to reconvert that into solar radii.
            #intens_proj = limbdark.sphere_intensity(body,lcdep)[1]/(c.Rsol)**2

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
        return {'abs_normal_intensities': abs_normal_intensities, 'normal_intensities': normal_intensities,
            'abs_intensities': abs_intensities, 'intensities': intensities,
            'ldint': ldint,
            'boost_factors': boost_factors}




class Envelope(Body):
    def __init__(self, Phi, masses, sma, ecc, freq_rot, teff1, teff2,
            abun, irrad_frac_refl1, irrad_frac_refl2, gravb_bol1, gravb_bol2, mesh_method='marching',
            dynamics_method='keplerian', mesh_init_phi=0.0, ind_self=0, ind_sibling=1, comp_no=1,
            atm='blackbody', datasets=[], passband={}, intens_weighting={},
            ld_func={}, ld_coeffs={},
            do_rv_grav=False, features=[], do_mesh_offset=True,
            label_envelope='contact_envelope', label_primary='primary',
            label_secondary='secondary', **kwargs):
        """
        [NOT IMPLEMENTED]

        :parameter float Phi: equipotential of this star at periastron
        :parameter masses: mass of each component in the system (solMass)
        :type masses: list of floats
        :parameter float sma: sma of this component's parent orbit (solRad)
        :parameter float abun: abundance of this star
        :parameter int ind_self: index in all arrays (positions, masses, etc) for the primary star in this contact_binary envelope
        :parameter int ind_sibling: index in all arrays (positions, masses, etc)
            for the secondary star in this contact_binary envelope
        :return: instantiated :class:`Envelope` object
        """
        super(Envelope, self).__init__(comp_no, ind_self, ind_sibling, masses,
                                       ecc, atm, datasets, passband,
                                       intens_weighting,
                                       ld_func, ld_coeffs,
                                       dynamics_method=dynamics_method,
                                       mesh_init_phi=mesh_init_phi)

        self.label_envelope = label_envelope
        self.label_primary = label_primary
        self.label_secondary = label_secondary

        # Remember how to compute the mesh
        self.mesh_method = mesh_method
        self.ntriangles = kwargs.get('ntriangles', 1000)                    # Marching
        self.distortion_method = kwargs.get('distortion_method', 'roche')   # Marching (WD assumes roche)
        self.gridsize = kwargs.get('gridsize', 90)                          # WD

        self.do_rv_grav = do_rv_grav

        # Remember things we need to know about this star - these will all be used
        # as defaults if they are not passed in future calls.  If for some reason
        # they are time dependent, then the instantaneous values need to be passed
        # for each call to update_position
        self.F = 1.0 # by definition for an contact_binary
        self.freq_rot = freq_rot   # TODO: change to just pass period and compute freq_rot here?
        self.sma = sma


        # compute Phi (Omega/pot): here again if we're the secondary star we have
        # to translate Phi since all meshing methods assume a primary component
        self.Phi_user = Phi  # this is the value set by the user (not translated)
        self._instantaneous_pot = Phi
        # for overcontacts, we'll always build the mesh from the primary star
        self.Phi = Phi

        self.teff1 = teff1
        self.teff2 = teff2

        self.irrad_frac_refl1 = irrad_frac_refl1
        self.irrad_frac_refl2 = irrad_frac_refl2
        self.gravb_bol1 = gravb_bol1
        self.gravb_bol2 = gravb_bol2
        # self.gravb_law = gravb_law

        # only putting this here so update_position doesn't complain
        self.irrad_frac_refl = 0.
        # self.gravb_law2 = gravb_law2


        # self.gravb_bol = gravb_bol
        # self.gravb_law = gravb_law
        self.abun = abun
        # self.irrad_frac_refl = irrad_frac_refl

        self.features = features  # TODO: move this to Body

        # Volume "conservation"
        self.volume_factor = 1.0  # TODO: eventually make this a parameter (currently defined to be the ratio between volumes at apastron/periastron)

        self._do_mesh_offset = do_mesh_offset

        # pblum scale needs to be different for envelopes - we need to actually
        # track the pblum per-component (envelope, primary, secondary) separately
        self._pblum_scale = {label_envelope: {},
                             label_primary: {},
                             label_secondary: {}}

        self._ptfarea      = {}


    @classmethod
    def from_bundle(cls, b, component, compute=None, dynamics_method='keplerian',
                    mesh_init_phi=0.0, datasets=[], **kwargs):
        """
        [NOT IMPLEMENTED]

        Build an contact_binary from the :class:`phoebe.frontend.bundle.Bundle` and its
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
            raise NotImplementedError("Contact envelope meshing requires a hierarchy to exist")


        label_envelope = component
        # self is just the primary star in the same orbit
        label_self = hier.get_sibling_of(component)  # TODO: make sure this defaults to primary
        label_sibling = hier.get_sibling_of(label_self)  # TODO: make sure this defaults to secondary
        label_orbit = hier.get_parent_of(component)
        # starrefs  = hier.get_stars()
        starrefs = hier.get_siblings_of(label_envelope)

        ind_self = starrefs.index(label_self)
        # for the sibling, we may need to handle a list of stars (ie in the case of a hierarchical triple)
        ind_sibling = starrefs.index(label_sibling) if isinstance(label_sibling, str) else [starrefs.index(l) for l in label_sibling]
        comp_no = 1

        # meshing for BRS needs d,q,F,Phi
        # d is instantaneous based on x,y,z of self and sibling
        # q is instantaneous based on masses of self and sibling
        # F we will assume is always 1 for an contact_binary
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
        #gravb_law = b.get_value('gravblaw_bol', component=component, context='component')
        #gravb_bol= b.get_value('gravb_bol', component=component, context='component')

        teff1 = b.get_value('teff', component=starrefs[0], context='component', unit=u.K)
        teff2 = b.get_value('teff', component=starrefs[1], context='component', unit=u.K)

        irrad_frac_refl1 = b.get_value('irrad_frac_refl_bol', component=starrefs[0], context='component')
        irrad_frac_refl2 = b.get_value('irrad_frac_refl_bol', component=starrefs[1], context='component')

        gravb_bol1 = b.get_value('gravb_bol', component=starrefs[0], context='component')
        gravb_bol2 = b.get_value('gravb_bol', component=starrefs[1], context='component')

        # gravb_law = b.get_value('gravblaw_bol', component=starrefs[0], context='component')
        #gravb_law2 = b.get_value('gravblaw_bol', component=starrefs[0], context='component')

        abun = b.get_value('abun', component=component, context='component')
        #frac_refl = b.get_value('frac_refl_bol', component=component, context='component')


        try:
            # TODO: will the rv_grav parameter ever be copied for the envelope?
            do_rv_grav = b.get_value('rv_grav', component=component, compute=compute, check_visible=False, **kwargs) if compute is not None else False
        except ValueError:
            # rv_grav may not have been copied to this component if no rvs are attached
            do_rv_grav = False

        # pass kwargs in case mesh_method was temporarily overridden
        # TODO: make sure mesh_method copies for envelopes
        mesh_method = b.get_value('mesh_method', component=component, compute=compute, **kwargs) if compute is not None else 'marching'

        mesh_kwargs = {}
        if mesh_method == 'marching':
            mesh_kwargs['ntriangles'] = b.get_value('ntriangles', component=component, compute=compute) if compute is not None else 1000
            mesh_kwargs['distortion_method'] = b.get_value('distortion_method', component=component, compute=compute) if compute is not None else 'roche'
        elif mesh_method == 'wd':
            mesh_kwargs['gridsize'] = b.get_value('gridsize', component=component, compute=compute) if compute is not None else 30
        else:
            raise NotImplementedError

        features = []
        # print "*** checking for features of", component, b.filter(component=component).features
        for feature in b.filter(component=component).features:
            # print "*** creating features", star, feature
            feature_ps = b.filter(feature=feature, component=component)
            feature_cls = globals()[feature_ps.kind.title()]
            features.append(feature_cls.from_bundle(b, feature))

        if conf.devel:
            do_mesh_offset = b.get_value('mesh_offset', compute=compute, **kwargs)
        else:
            do_mesh_offset = True


        datasets_intens = [ds for ds in b.filter(kind=['lc', 'rv', 'ifm'], context='dataset').datasets if ds != '_default']
        atm = b.get_value('atm', compute=compute, component=component, **kwargs) if compute is not None else 'blackbody'
        passband = {ds: b.get_value('passband', dataset=ds, **kwargs) for ds in datasets_intens}
        intens_weighting = {ds: b.get_value('intens_weighting', dataset=ds, **kwargs) for ds in datasets_intens}
        ld_func = {ds: b.get_value('ld_func', dataset=ds, component=component, **kwargs) for ds in datasets_intens}
        ld_coeffs = {ds: b.get_value('ld_coeffs', dataset=ds, component=component, check_visible=False, **kwargs) for ds in datasets_intens}
        ld_func['bol'] = b.get_value('ld_func_bol', component=component, context='component', **kwargs)
        ld_coeffs['bol'] = b.get_value('ld_coeffs_bol', component=component, context='component', **kwargs)

        return cls(Phi, masses, sma, ecc, freq_rot, teff1, teff2, abun, irrad_frac_refl1, irrad_frac_refl2,
                gravb_bol1, gravb_bol2, mesh_method, dynamics_method,
                mesh_init_phi, ind_self, ind_sibling, comp_no,
                atm=atm,
                datasets=datasets, passband=passband,
                intens_weighting=intens_weighting,
                ld_func=ld_func, ld_coeffs=ld_coeffs,
                do_rv_grav=do_rv_grav,
                features=features, do_mesh_offset=do_mesh_offset,
                label_envelope=label_envelope,
                label_primary=label_self, label_secondary=label_sibling,
                **mesh_kwargs)

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
        # TODO: should F be fixed at 1 - is this the job of the frontend or backend?

        Phi = kwargs.get('Phi', self.Phi)  # NOTE: self.Phi automatically corrects for the secondary star
        q = self.q  # NOTE: this is automatically flipped to be 1./q for secondary components

        mesh_args = (q, F, d, Phi)

        if mesh_method == 'marching':
            # Phi = kwargs.get('Phi', self.Phi_user)  # NOTE: self.Phi_user is not corrected for the secondary star, but that's fine because we pass primary vs secondary as choice
            # q = 1./self.q if self.comp_no == 2 else self.q  # NOTE: undo the inversion so this is ALWAYS Mp/Ms

            ntriangles = kwargs.get('ntriangles', self.ntriangles)


            if self.distortion_method == 'roche':
                # TODO: check whether roche or misaligned roche from values of incl, etc!!!!

                av = libphoebe.roche_area_volume(*mesh_args,
                                                 choice=2,
                                                 larea=True,
                                                 lvolume=True)

                delta = np.sqrt(4./np.sqrt(3) * av['larea'] / ntriangles)

                new_mesh = libphoebe.roche_marching_mesh(*mesh_args,
                                                         delta=delta,
                                                         choice=2,
                                                         full=True,
                                                         max_triangles=ntriangles*2,
                                                         vertices=True,
                                                         triangles=True,
                                                         centers=True,
                                                         vnormals=True,
                                                         tnormals=True,
                                                         cnormals=False,
                                                         vnormgrads=True,
                                                         cnormgrads=False,
                                                         areas=True,
                                                         volume=False)


                # Now we'll get the area and volume of the Roche potential
                # itself (not the mesh).
                # TODO: which volume(s) do we want to report?  Either way, make
                # sure to do the same for the OC case and rotstar
                new_mesh['volume'] = av['lvolume']

                if self._do_mesh_offset:
                    # vertices directly from meshing are placed directly on the
                    # potential, causing the volume and surface area to always
                    # (for convex surfaces) be underestimated.  Now let's jitter
                    # each of the vertices along their normals to recover the
                    # expected volume/surface area.  Since they are moved along
                    # their normals, vnormals applies to both vertices and
                    # pvertices.
                    new_mesh['pvertices'] = new_mesh.pop('vertices')
                    mo = libphoebe.mesh_offseting(av['larea'],
                                                  new_mesh['pvertices'],
                                                  new_mesh['vnormals'],
                                                  new_mesh['triangles'],
                                                  curvature=True,
                                                  vertices=True,
                                                  tnormals=True,
                                                  areas=True,
                                                  volume=False)

                    new_mesh['vertices'] = mo['vertices']
                    new_mesh['areas'] = mo['areas']
                    new_mesh['tnormals'] = mo['tnormals']

                    # TODO: need to update centers (so that they get passed
                    # to the frontend as x, y, z)
                    # new_mesh['centers'] = mo['centers']


                else:
                    # pvertices should just be a copy of vertice
                    new_mesh['pvertices'] = new_mesh['vertices']

                # We only need the gradients where we'll compute local
                # quantities which, for a marching mesh, is at the vertices.
                new_mesh['normgrads'] = new_mesh.pop('vnormgrads')

                # And lastly, let's fill the velocities column - with zeros
                # at each of the vertices
                new_mesh['velocities'] = np.zeros(new_mesh['vertices'].shape)

                new_mesh['tareas'] = np.array([])

                # WD style overcontacts require splitting of the mesh into two components
                # env_comp = 0 for primary part of the envelope, 1 for secondary

                # compute the positions of the minimum radii of the neck in the xy and xz planes
                # when temperature_method becomes available, wrap this with if tmethod='wd':
                L1 = potentials.Lag1(q)
                xz, z = potentials.nekmin(Phi, q, L1, 0.05)
                # choose which value of x to use as the minimum (maybe extend to average of both?
                xmin = xz

                # create the env_comp array and change the values of all where vertices x>xmin to 1
                env_comp = np.zeros(len(new_mesh['vertices']))

                env_comp[new_mesh['vertices'][:,0]>xmin] = 1
                #
                new_mesh['env_comp'] = env_comp
                # print new_mesh['env_comp']

                # do the similar for triangles
                env_comp3 = np.zeros(len(new_mesh['triangles']))

                # Uncomment this is we want to average over vertices and comment below :/
                # for i in range(len(new_mesh['triangles'])):
                #
                #     #take the vertex indices of each triangle
                #     vind = new_mesh['triangles'][i]
                #     print 'vind: ', vind
                #     env_comp3[i] = np.average([new_mesh['env_comp'][vind[0]],new_mesh['env_comp'][vind[1]],new_mesh['env_comp'][vind[2]]])
                #

                new_mesh['env_comp3']=env_comp3

                # Comment this is we want to average over vertices
                N = len(new_mesh['vertices'])
                for i in range(len(new_mesh['triangles'])):

                    #take the vertex indices of each triangle
                    vind = new_mesh['triangles'][i]
                    center = new_mesh['centers'][i]

                    # the adding of vertices and vertex parameters should go in a function

                    def add_vertex_to_mesh(i,j,vind,comp):

                        N = len(new_mesh['vertices'])
                        # add vertex params
                        new_mesh['vertices'] = np.vstack((new_mesh['vertices'],new_mesh['vertices'][vind]))
                        new_mesh['pvertices'] = np.vstack((new_mesh['pvertices'],new_mesh['pvertices'][vind]))
                        new_mesh['vnormals'] = np.vstack((new_mesh['vnormals'],new_mesh['vnormals'][vind]))
                        new_mesh['normgrads'] = np.hstack((new_mesh['normgrads'],new_mesh['normgrads'][vind]))
                        new_mesh['velocities'] = np.vstack((new_mesh['velocities'],np.zeros(3)))
                        new_mesh['env_comp'] = np.hstack((new_mesh['env_comp'],comp))
                        new_mesh['triangles'][i][j] = N

                    if center[0] <= xmin:
                        new_mesh['env_comp3'][i] = 0
                        if new_mesh['vertices'][vind[0]][0] > xmin:
                            add_vertex_to_mesh(i,0,vind[0],0)
                        else:
                            new_mesh['env_comp'][vind[0]] = 0

                        if new_mesh['vertices'][vind[1]][0] > xmin:
                            add_vertex_to_mesh(i,1,vind[1],0)
                        else:
                            new_mesh['env_comp'][vind[1]] = 0

                        if new_mesh['vertices'][vind[2]][0] > xmin:
                            add_vertex_to_mesh(i,2,vind[2],0)
                        else:
                            new_mesh['env_comp'][vind[2]] = 0

                    else:
                        new_mesh['env_comp3'][i] = 1
                        if new_mesh['vertices'][vind[0]][0] <= xmin:
                            add_vertex_to_mesh(i,0,vind[0],1)
                        else:
                            new_mesh['env_comp'][vind[0]] = 1

                        if new_mesh['vertices'][vind[1]][0] <=xmin:
                            add_vertex_to_mesh(i,1,vind[1],1)
                        else:
                            new_mesh['env_comp'][vind[1]] = 1

                        if new_mesh['vertices'][vind[2]][0] <= xmin:
                            add_vertex_to_mesh(i,2,vind[2],1)
                        else:
                            new_mesh['env_comp'][vind[2]] = 1

                # compute fractional areas of vertices

                # new_mesh['frac_areas']=potentials.compute_frac_areas(new_mesh,xmin)

            elif self.distortion_method == 'nbody':
                # TODO: implement this? - can OCs be done in NBody mode?
                raise NotImplementedError("'nbody' distortion_method not yet supported - try roche")
            else:
                raise NotImplementedError

        elif mesh_method == 'wd':

            N = int(kwargs.get('gridsize', self.gridsize))

            the_grid = potentials.discretize_wd_style_oc(N, *mesh_args)
            new_mesh = mesh.wd_grid_to_mesh_dict(the_grid, q, F, d)
            scale = sma

            # WD style overcontacts require splitting of the mesh into two components
            # env_comp = 0 for primary part of the envelope, 1 for secondary

            # compute the positions of the minimum radii of the neck in the xy and xz planes
            xy,xz,y,z = potentials.nekmin(Phi,q,0.5,0.05,0.05)
            # choose which value of x to use as the minimum (maybe extend to average of both?
            xmin = xz

            # create the env_comp array and change the values of all where vertices x>xmin to 1
            env_comp = np.zeros(len(new_mesh['centers']))
            env_comp[new_mesh['centers'][:,0]>xmin] = 1

            new_mesh['env_comp'] = env_comp
            new_mesh['env_comp3']=env_comp

        else:
            raise NotImplementedError("mesh_method '{}' is not supported".format(mesh_method))


        new_mesh['label_envelope'] = self.label_envelope
        new_mesh['label_primary'] = self.label_primary
        new_mesh['label_secondary'] = self.label_secondary

        return new_mesh, sma, mesh_args


    def _compute_instantaneous_quantities(self, xs, ys, zs, **kwargs):
        """
        TODO: add documentation
        """
        pole_func = getattr(libphoebe, '{}_pole'.format(self.distortion_method))
        gradOmega_func = getattr(libphoebe, '{}_gradOmega_only'.format(self.distortion_method))

        r_pole1 = pole_func(*self._mesh_args,choice=0)
        r_pole2 = pole_func(*self._mesh_args,choice=1)

        r_pole1_ = np.array([0., 0., r_pole1])
        r_pole2_ = np.array([0., 0., r_pole2])

        args1 = list(self._mesh_args)[:-1]+[r_pole1_]
        args2 = list(self._mesh_args)[:-1]+[r_pole2_]

        grads1 = gradOmega_func(*args1)
        grads2 = gradOmega_func(*args2)

        g_pole1 = np.linalg.norm(grads1)
        g_pole2 = np.linalg.norm(grads2)

        g_rel_to_abs = c.G.si.value*c.M_sun.si.value*self.masses[self.ind_self]/(self.sma*c.R_sun.si.value)**2*100. # 100 for m/s**2 -> cm/s**2

        self._instantaneous_gpole1 = g_pole1 * g_rel_to_abs
        self._instantaneous_gpole2 = g_pole2 * g_rel_to_abs
        # TODO NOW: check whether r_pole is in absolute units (scaled/not scaled)
        self._instantaneous_rpole1 = r_pole1
        self._instantaneous_rpole2 = r_pole2

    def _fill_loggs(self, mesh=None, ignore_effects=False):
        """
        TODO: add documentation

        Calculate local surface gravity

        GMSunNom = 1.3271244e20 m**3 s**-2
        RSunNom = 6.597e8 m
        """

        if mesh is None:
            mesh = self.mesh

        g_rel_to_abs = c.G.si.value*c.M_sun.si.value*self.masses[self.ind_self]/(self.sma*c.R_sun.si.value)**2*100. # 100 for m/s**2 -> cm/s**2

        loggs = np.log10(mesh.normgrads.for_computations * g_rel_to_abs)

        if not ignore_effects:
            for feature in self.features:
                if feature.proto_coords:
                    teffs = feature.process_loggs(loggs, self.get_standard_mesh().coords_for_computations, t=self.time)
                else:
                    teffs = feature.process_loggs(loggs, mesh.coords_for_computations, t=self.time)

        mesh.update_columns(loggs=loggs)


    def _fill_gravs(self, mesh=None, **kwargs):
        """
        TODO: add documentation

        requires _fill_loggs to have been called
        """
        if mesh is None:
            mesh = self.mesh


        # TODO: rename 'gravs' to 'gdcs' (gravity darkening corrections)

        g_rel_to_abs = c.G.si.value*c.M_sun.si.value*self.masses[self.ind_self]/(self.sma*c.R_sun.si.value)**2*100. # 100 for m/s**2 -> cm/s**2
        # TODO: check the division by 100 - is this just to change units back to m?
        gravs1 = ((mesh.normgrads.for_computations[mesh.env_comp==0] * g_rel_to_abs)/self._instantaneous_gpole1)**self.gravb_bol1
        gravs2 = ((mesh.normgrads.for_computations[mesh.env_comp==1] * g_rel_to_abs)/self._instantaneous_gpole2)**self.gravb_bol2

        # TODO: make sure equivalent to the old way here
        # gravs = abs(10**(self.mesh.loggs.for_computations-2)/self._instantaneous_gpole)**self.gravb_bol
        gravs = np.zeros(len(mesh.env_comp))
        gravs[mesh.env_comp==0]=gravs1
        gravs[mesh.env_comp==1]=gravs2

        mesh.update_columns(gravs=gravs)

    def _fill_teffs(self, mesh=None, ignore_effects=False, **kwargs):
        r"""

        requires _fill_loggs and _fill_gravs to have been called

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
        if mesh is None:
            mesh = self.mesh

        Teff1 = kwargs.get('teff1', self.teff1)
        Teff2 = kwargs.get('teff2', self.teff2)

        # Convert from mean to polar by dividing total flux by gravity darkened flux (Ls drop out)
        Tpole1 = Teff1*(np.sum(mesh.areas[mesh.env_comp3==0]) / np.sum(mesh.gravs.centers[mesh.env_comp3==0]*mesh.areas[mesh.env_comp3==0]))**(0.25)
        Tpole2 = Teff2*(np.sum(mesh.areas[mesh.env_comp3==1]) / np.sum(mesh.gravs.centers[mesh.env_comp3==1]*mesh.areas[mesh.env_comp3==1]))**(0.25)

        self._instantaneous_teffpole1 = Tpole1
        self._instantaneous_teffpole2 = Tpole2

        # Now we can compute the local temperatures.
        teffs1 = (mesh.gravs.for_computations[mesh.env_comp==0] * Tpole1**4)**0.25
        teffs2 = (mesh.gravs.for_computations[mesh.env_comp==1] * Tpole2**4)**0.25

        if not ignore_effects:
            for feature in self.features:
                if feature.proto_coords:
                    teffs1 = feature.process_teffs(teffs, self.get_standard_mesh().coords_for_computations[mesh.env_comp==0], t=self.time)
                    teffs2 = feature.process_teffs(teffs, self.get_standard_mesh().coords_for_computations[mesh.env_comp==1], t=self.time)
                else:
                    teffs1 = feature.process_teffs(teffs, mesh.coords_for_computations[mesh.env_comp==0], t=self.time)
                    teffs2 = feature.process_teffs(teffs, mesh.coords_for_computations[mesh.env_comp==1], t=self.time)

        teffs = np.zeros(len(mesh.env_comp))
        teffs[mesh.env_comp==0]=teffs1
        teffs[mesh.env_comp==1]=teffs2

        mesh.update_columns(teffs=teffs)

    def _fill_albedos(self, mesh=None, irrad_frac_refl=0.0):
        """
        TODO: add documentation
        """
        if mesh is None:
            mesh = self.mesh
            irrad_frac_refl1 = self.irrad_frac_refl1
            irrad_frac_refl2 = self.irrad_frac_refl2

        irrad_frac_refl = np.zeros(len(mesh.env_comp))
        irrad_frac_refl[mesh.env_comp==0] = irrad_frac_refl1
        irrad_frac_refl[mesh.env_comp==1] = irrad_frac_refl2

        mesh.update_columns(irrad_frac_refl=irrad_frac_refl)

    def compute_luminosity(self, dataset, component=None, **kwargs):
        """
        """

        if component is None:
            component = self.label_envelope

        # areas are the NON-projected areas of each surface element.  We'll be
        # integrating over normal intensities, so we don't need to worry about
        # multiplying by mu to get projected areas.
        areas = self.mesh.areas_si

        # abs_normal_intensities are directly out of the passbands module and are
        # emergent normal intensities in this dataset's passband/atm in absolute units
        abs_normal_intensities = self.mesh['abs_normal_intensities:{}'.format(dataset)].centers

        ldint = self.mesh['ldint:{}'.format(dataset)].centers
        ptfarea = self.get_ptfarea(dataset)

        if component == self.label_envelope:
            areas = areas
            abs_normal_intensities = abs_normal_intensities
            ldint = ldint
        elif component == self.label_primary:
            areas = areas[self.mesh.env_comp3 == 0]
            abs_normal_intensities = abs_normal_intensities[self.mesh.env_comp3 == 0]
            ldint = ldint[self.mesh.env_comp3 == 0]
        elif component == self.label_secondary:
            areas = areas[self.mesh.env_comp3 == 1]
            abs_normal_intensities = abs_normal_intensities[self.mesh.env_comp3 == 1]
            ldint = ldint[self.mesh.env_comp3 == 1]
        else:
            raise ValueError

        # Our total integrated intensity in absolute units (luminosity) is now
        # simply the sum of the normal emergent intensities times pi (to account
        # for intensities emitted in all directions across the solid angle),
        # limbdarkened as if they were at mu=1, and multiplied by their respective areas

        abs_luminosity = np.sum(abs_normal_intensities*areas*ldint)*ptfarea*np.pi

        # NOTE: when this is computed the first time (for the sake of determining
        # pblum_scale), get_pblum_scale will return 1.0
        return abs_luminosity * self.get_pblum_scale(dataset)


    def compute_pblum_scale(self, dataset, pblum, component=None, **kwargs):
        """
        intensities should already be computed for this dataset at the time for which pblum is being provided

        TODO: add documentation
        """
        if component is None:
            component = self.label_envelope

        total_integrated_intensity = self.compute_luminosity(dataset, component=component, **kwargs)
        # print 'Total integrated luminosity for component %s is %d.' % (component, total_integrated_intensity)
        # We now want to remember the scale for all intensities such that the
        # luminosity in relative units gives the provided pblum
        pblum_scale = pblum / total_integrated_intensity
        # print 'Pblum scale for component %s is %d.' % (component, pblum_scale)
        # self._pblum_scale[component][dataset] = pblum_scale
        self.set_pblum_scale(dataset, component=component, pblum_scale=pblum_scale)

    def set_pblum_scale(self, dataset, pblum_scale, component=None, **kwargs):
        if component is None:
            component = self.label_envelope

        self._pblum_scale[component][dataset] = pblum_scale

    def get_pblum_scale(self, dataset, component=None):
        """
        """
        if component is None:
            component = self.label_envelope

        if dataset in self._pblum_scale[component].keys():
            return self._pblum_scale[component][dataset]
        else:
            return 1.

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

        # We need to fill all the flux-related columns so that we can weigh each
        # triangle's rv by its flux in the requested passband.
        lc_cols = self._populate_lc(dataset, **kwargs)

        # rv per element is just the z-component of the velocity vectory.  Note
        # the change in sign from our right-handed system to rv conventions.
        # These will be weighted by the fluxes when integrating

        rvs = -1*self.mesh.velocities.for_computations[:,2]


        # Gravitational redshift
        if self.do_rv_grav:
            rv_grav = c.G*(self.mass*u.solMass)/(self._instantaneous_rpole*u.solRad)/c.c
            # rvs are in solrad/d internally
            rv_grav = rv_grav.to('solRad/d').value

            rvs += rv_grav

        cols = lc_cols
        cols['rvs'] = rvs
        return cols

    def _populate_lc(self, dataset,  **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`

        :raises NotImplementedError: if lc_method is not supported
        """

        lc_method = kwargs.get('lc_method', 'numerical')  # TODO: make sure this is actually passed

        passband = kwargs.get('passband', self.passband.get(dataset, None))
        intens_weighting = kwargs.get('intens_weighting', self.intens_weighting.get(dataset, None))
        ld_func = kwargs.get('ld_func', self.ld_func.get(dataset, None))
        ld_coeffs = kwargs.get('ld_coeffs', self.ld_coeffs.get(dataset, None)) if ld_func != 'interp' else None
        atm = kwargs.get('atm', self.atm)
        boosting_method = kwargs.get('boosting_method', self.boosting_method)

        pblum = kwargs.get('pblum', 4*np.pi)


        if lc_method=='numerical':

            pb = passbands.get_passband(passband)

            ptfarea = pb.wl[-1]-pb.wl[0]
            self.set_ptfarea(dataset, ptfarea)

            ldint = pb.ldint(Teff=self.mesh.teffs.for_computations,
                             logg=self.mesh.loggs.for_computations,
                             abun=self.mesh.abuns.for_computations,
                             atm=atm,
                             ld_func=ld_func,
                             ld_coeffs=ld_coeffs,
                             photon_weighted=intens_weighting=='photon')

            # abs_normal_intensities are the normal emergent passband intensities:
            abs_normal_intensities = pb.Inorm(Teff=self.mesh.teffs.for_computations,
                                              logg=self.mesh.loggs.for_computations,
                                              abun=self.mesh.abuns.for_computations,
                                              atm=atm,
                                              ldint=ldint,
                                              photon_weighted=intens_weighting=='photon')


            # abs_intensities are the projected (limb-darkened) passband intensities
            # TODO: why do we need to use abs(mus) here?
            abs_intensities = pb.Imu(Teff=self.mesh.teffs.for_computations,
                                     logg=self.mesh.loggs.for_computations,
                                     abun=self.mesh.abuns.for_computations,
                                     mu=abs(self.mesh.mus_for_computations),
                                     atm=atm,
                                     ldint=ldint,
                                     ld_func=ld_func,
                                     ld_coeffs=ld_coeffs,
                                     photon_weighted=intens_weighting=='photon')


            # Beaming/boosting
            if boosting_method == 'none':
                boost_factors = 1.0
            elif boosting_method == 'linear':
                bindex = pb.bindex(Teff=self.mesh.teffs.for_computations,
                                   logg=self.mesh.loggs.for_computations,
                                   abun=self.mesh.abuns.for_computations,
                                   mu=abs(self.mesh.mus_for_computations),
                                   atm=atm,
                                   photon_weighted=intens_weighting=='photon')

                boost_factors = 1.0 + bindex * self.mesh.velocities.for_computations[:,2]/37241.94167601236
            else:
                raise NotImplementedError("boosting_method='{}' not supported".format(self.boosting_method))

            # TODO: does this make sense to boost proj but not norm?
            abs_intensities *= boost_factors

            # Handle pblum - distance and l3 scaling happens when integrating (in observe)
            # we need to scale each triangle so that the summed normal_intensities over the
            # entire star is equivalent to pblum / 4pi

            normal_intensities = abs_normal_intensities * self.get_pblum_scale(dataset,component=self.label_primary)
            intensities = abs_intensities * self.get_pblum_scale(dataset,component=self.label_primary)

            # print 'primary pblum scale', self.get_pblum_scale(dataset,component=self.label_primary)

            normal_intensities[self.mesh.env_comp==1] = abs_normal_intensities[self.mesh.env_comp==1] * self.get_pblum_scale(dataset,component=self.label_secondary)
            intensities[self.mesh.env_comp==1] = abs_intensities[self.mesh.env_comp==1] * self.get_pblum_scale(dataset,component=self.label_secondary)

            # print 'secondary pblum scale', self.get_pblum_scale(dataset,component=self.label_secondary)

        elif lc_method=='analytical':
            raise NotImplementedError("analytical fluxes not yet ported to beta")
            #lcdep, ref = system.get_parset(ref)
            # The projected intensity is normalised with the distance in cm, we need
            # to reconvert that into solar radii.
            #intens_proj = limbdark.sphere_intensity(body,lcdep)[1]/(c.Rsol)**2

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
        return {'abs_normal_intensities': abs_normal_intensities, 'normal_intensities': normal_intensities,
            'abs_intensities': abs_intensities, 'intensities': intensities,
            'ldint': ldint,
            'boost_factors': boost_factors}

class Feature(object):
    """
    Note that for all features, each of the methods below will be called.  So
    changing the coordinates WILL affect the original/intrinsic loggs which
    will then be used as input for that method call.

    In other words, its probably safest if each feature only overrides a
    SINGLE one of the methods.  Overriding multiple methods should be done
    with great care.
    """
    def __init__(self, *args, **kwargs):
        pass

    @property
    def proto_coords(self):
        """
        Override this to True if all methods (except process_coords*... those
        ALWAYS expect protomesh coordinates) are expecting coordinates
        in the protomesh (star) frame-of-reference rather than the
        current in-orbit system frame-of-reference.
        """
        return False

    def process_coords_for_computations(self, coords_for_computations, t):
        """
        Method for a feature to process the coordinates.  Coordinates are
        processed AFTER scaling but BEFORE being placed in orbit.

        NOTE: coords_for_computations affect physical properties only and
        not geometric properties (areas, eclipse detection, etc).  If you
        want to override geometric properties, use the hook for
        process_coords_for_observations as well.

        Features that affect coordinates_for_computations should override
        this method
        """
        return coords_for_computations

    def process_coords_for_observations(self, coords_for_computations, coords_for_observations, t):
        """
        Method for a feature to process the coordinates.  Coordinates are
        processed AFTER scaling but BEFORE being placed in orbit.

        NOTE: coords_for_observations affect the geometry only (areas of each
        element and eclipse detection) but WILL NOT affect any physical
        parameters (loggs, teffs, intensities).  If you want to override
        physical parameters, use the hook for process_coords_for_computations
        as well.

        Features that affect coordinates_for_observations should override this method.
        """
        return coords_for_observations

    def process_loggs(self, loggs, coords, t=None):
        """
        Method for a feature to process the loggs.

        Features that affect loggs should override this method
        """
        return loggs

    def process_teffs(self, teffs, coords, t=None):
        """
        Method for a feature to process the teffs.

        Features that affect teffs should override this method
        """
        return teffs

class Spot(Feature):
    def __init__(self, colat, longitude, dlongdt, radius, relteff, t0, **kwargs):
        """
        Initialize a Spot feature
        """
        super(Spot, self).__init__(**kwargs)
        self._colat = colat
        self._longitude = longitude
        self._radius = radius
        self._relteff = relteff
        self._dlongdt = dlongdt
        self._t0 = t0

    @classmethod
    def from_bundle(cls, b, feature):
        """
        Initialize a Spot feature from the bundle.
        """

        feature_ps = b.get_feature(feature)

        colat = feature_ps.get_value('colat', unit=u.rad)
        longitude = feature_ps.get_value('long', unit=u.rad)

        if len(b.hierarchy.get_stars())>=2:
            star_ps = b.get_component(feature_ps.component)
            orbit_ps = b.get_component(b.hierarchy.get_parent_of(feature_ps.component))
            syncpar = star_ps.get_value('syncpar')
            period = orbit_ps.get_value('period')
            dlongdt = (syncpar - 1) / period * 2 * np.pi
        else:
            star_ps = b.get_component(feature_ps.component)
            dlongdt = star_ps.get_value('freq', unit=u.rad/u.d)
            longitude = np.pi/2

        radius = feature_ps.get_value('radius', unit=u.rad)
        relteff = feature_ps.get_value('relteff', unit=u.dimensionless_unscaled)

        t0 = b.get_value('t0', context='system', unit=u.d)

        return cls(colat, longitude, dlongdt, radius, relteff, t0)

    @property
    def proto_coords(self):
        """
        """
        return True

    def pointing_vector(self, time):
        t = time - self._t0
        x = np.sin(self._colat)*np.cos(self._longitude + self._dlongdt * t)
        y = np.sin(self._colat)*np.sin(self._longitude + self._dlongdt * t)
        z = np.cos(self._colat)
        return np.array([x,y,z])

    def process_teffs(self, teffs, coords, t=None):
        """
        Change the local effective temperatures for any values within the
        "cone" defined by the spot.  Any teff within the spot will have its
        current value multiplied by the "relteff" factor

        :parameter array teffs: array of teffs for computations
        :parameter array coords: array of coords for computations
        :t float: current time
        """
        if t is None:
            # then assume at t0
            t = self._t0

        cos_alpha_coords = np.dot(coords, self.pointing_vector(t)) / np.linalg.norm(coords, axis=1)
        cos_alpha_spot = np.cos(self._radius)

        filter_ = cos_alpha_coords > cos_alpha_spot

        teffs[filter_] = teffs[filter_] * self._relteff

        return teffs

class Pulsation(Feature):
    def __init__(self, radamp, freq, l=0, m=0, tanamp=0.0, teffext=False, **kwargs):
        self._freq = freq
        self._radamp = radamp
        self._l = l
        self._m = m
        self._tanamp = tanamp

        self._teffext = teffext

    @classmethod
    def from_bundle(cls, b, feature):
        """
        Initialize a Pulsation feature from the bundle.
        """

        feature_ps = b.get_feature(feature)
        freq = feature_ps.get_value('freq', unit=u.d**-1)
        radamp = feature_ps.get_value('radamp', unit=u.dimensionless_unscaled)
        l = feature_ps.get_value('l', unit=u.dimensionless_unscaled)
        m = feature_ps.get_value('m', unit=u.dimensionless_unscaled)
        teffext = feature_ps.get_value('teffext')

        GM = c.G.to('solRad3 / (solMass d2)').value*b.get_value(qualifier='mass', component=feature_ps.component, context='component', unit=u.solMass)
        R = b.get_value(qualifier='rpole', component=feature_ps.component, section='component', unit=u.solRad)

        tanamp = GM/R**3/freq**2

        return cls(radamp, freq, l, m, tanamp, teffext)

    @property
    def proto_coords(self):
        """
        """
        return True

    def dYdtheta(self, m, l, theta, phi):
        if abs(m) > l:
            return 0

        # TODO: just a quick hack
        if abs(m+1) > l:
            last_term = 0.0
        else:
            last_term = Y(m+1, l, theta, phi)

        return m/np.tan(theta)*Y(m, l, theta, phi) + np.sqrt((l-m)*(l+m+1))*np.exp(-1j*phi)*last_term

    def dYdphi(self, m, l, theta, phi):
        return 1j*m*Y(m, l, theta, phi)

    def process_coords_for_computations(self, coords_for_computations, t):
        """
        """
        if self._teffext:
            return coords_for_computations

        x, y, z, r = coords_for_computations[:,0], coords_for_computations[:,1], coords_for_computations[:,2], np.sqrt((coords_for_computations**2).sum(axis=1))
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)

        xi_r = self._radamp * Y(self._m, self._l, theta, phi) * np.exp(-1j*2*np.pi*self._freq*t)
        xi_t = self._tanamp * self.dYdtheta(self._m, self._l, theta, phi) * np.exp(-1j*2*np.pi*self._freq*t)
        xi_p = self._tanamp/np.sin(theta) * self.dYdphi(self._m, self._l, theta, phi) * np.exp(-1j*2*np.pi*self._freq*t)

        new_coords = np.zeros(coords_for_computations.shape)
        new_coords[:,0] = coords_for_computations[:,0] + xi_r * np.sin(theta) * np.cos(phi)
        new_coords[:,1] = coords_for_computations[:,1] + xi_r * np.sin(theta) * np.sin(phi)
        new_coords[:,2] = coords_for_computations[:,2] + xi_r * np.cos(theta)

        return new_coords

    def process_coords_for_observations(self, coords_for_computations, coords_for_observations, t):
        """
        Displacement equations:

          xi_r(r, theta, phi)     = a(r) Y_lm (theta, phi) exp(-i*2*pi*f*t)
          xi_theta(r, theta, phi) = b(r) dY_lm/dtheta (theta, phi) exp(-i*2*pi*f*t)
          xi_phi(r, theta, phi)   = b(r)/sin(theta) dY_lm/dphi (theta, phi) exp(-i*2*pi*f*t)

        where:

          b(r) = a(r) GM/(R^3*f^2)
        """
        # TODO: we do want to displace the coords_for_observations, but the x,y,z,r below are from the ALSO displaced coords_for_computations
        # if not self._teffext:
            # return coords_for_observations

        x, y, z, r = coords_for_computations[:,0], coords_for_computations[:,1], coords_for_computations[:,2], np.sqrt((coords_for_computations**2).sum(axis=1))
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)

        xi_r = self._radamp * Y(self._m, self._l, theta, phi) * np.exp(-1j*2*np.pi*self._freq*t)
        xi_t = self._tanamp * self.dYdtheta(self._m, self._l, theta, phi) * np.exp(-1j*2*np.pi*self._freq*t)
        xi_p = self._tanamp/np.sin(theta) * self.dYdphi(self._m, self._l, theta, phi) * np.exp(-1j*2*np.pi*self._freq*t)

        new_coords = np.zeros(coords_for_observations.shape)
        new_coords[:,0] = coords_for_observations[:,0] + xi_r * np.sin(theta) * np.cos(phi)
        new_coords[:,1] = coords_for_observations[:,1] + xi_r * np.sin(theta) * np.sin(phi)
        new_coords[:,2] = coords_for_observations[:,2] + xi_r * np.cos(theta)

        return new_coords

    def process_teffs(self, teffs, coords, t=None):
        """
        """
        if not self._teffext:
            return teffs

        raise NotImplementedError("teffext=True not yet supported for pulsations")
