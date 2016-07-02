import numpy as np
from scipy.optimize import newton
from math import sqrt, sin, cos, acos, atan2, trunc, pi
import os
import copy

from phoebe.atmospheres import limbdark, passbands
from phoebe.distortions import roche, rotstar
from phoebe.backend import eclipse, potentials, mesh
import libphoebe

from phoebe import u
from phoebe import c

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
    def __init__(self, bodies_dict, eclipse_alg='graham', dynamics_method='keplerian'):
        """
        :parameter dict bodies_dict: dictionary of component names and Bodies (or subclass of Body)
        """
        self._bodies = bodies_dict
        self.eclipse_alg = eclipse_alg
        # self.subdiv_alg = subdiv_alg
        # self.subdiv_num = subdiv_num
        self.dynamics_method = dynamics_method

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
                compute_ps = b.get_compute(compute)
            else:
                # then hopefully compute is the parameterset
                compute_ps = compute
            eclipse_alg = compute_ps.get_value(qualifier='eclipse_alg', **kwargs)
            # subdiv_alg = 'edge' #compute_ps.get_value(qualifier='subdiv_alg', **kwargs)
            # subdiv_num = compute_ps.get_value(qualifier='subdiv_num', **kwargs)
            dynamics_method = compute_ps.get_value(qualifier='dynamics_method', **kwargs)
        else:
            eclipse_alg = 'graham'
            # subdiv_alg = 'edge'
            # subdiv_num = 3
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
        # meshes.update_column(visibilities=visibilities)
        # meshes.update_column(somethingelse=somethingelse)
        #
        # rather than calling self.meshes repeatedly

        return mesh.Meshes(self._bodies)


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
            eclipse_alg = 'only_horizon'

        # meshes is an object which allows us to easily access and update columns
        # in the meshes *in memory*.  That is meshes.update_columns will propogate
        # back to the current mesh for each body.
        meshes = self.meshes

        # Reset all visibilities to be fully visible to start
        meshes.update_columns('visiblities', 1.0)

        ecl_func = getattr(eclipse, eclipse_alg)

        # TODO: FOR TESTING ONLY - EITHER REMOVE THIS OR MAKE IT LESS HACKY!!!
        # if eclipse_alg == 'wd_horizon':
        #     import phoebeBackend as phb1
        #     from phoebe import io

        #     if not hasattr(self, '_phb1_init'):
        #         phb1.init()
        #         try:
        #             phb1.configure()
        #         except SystemError:
        #             raise SystemError("PHOEBE config failed: try creating PHOEBE config file through GUI")
        #         self._phb1_init = True

        #     phb1.open('_tmp_legacy_inp')
        #     stuff = phb1.lc((self.time,), 0, 1)

        # We need to run eclipse detection first to get the partial triangles
        # to send to subdivision
        visibilities, weights = ecl_func(meshes, self.xs, self.ys, self.zs)
        # visiblilities here is a dictionary with keys being the component
        # labels and values being the np arrays of visibilities.  We can pass
        # this dictionary directly and the columns will be applied respectively.
        meshes.update_columns('visibilities', visibilities)
        if weights is not None:
            meshes.update_columns('weights', weights)

        return


    def observe(self, dataset, method, components=None, distance=1.0, l3=0.0):
        """
        TODO: add documentation

        Integrate over visible surface elements and return a dictionary of observable values

        distance (m)
        """

        meshes = self.meshes
        if method=='RV':
            visibilities = meshes.get_column_flat('visibilities', components)

            if np.all(visibilities==0):
                # then no triangles are visible, so we should return nan
                #print "NO VISIBLE TRIANGLES!!!"
                return {'rv': np.nan}

            rvs = meshes.get_column_flat("rv:{}".format(dataset), components)
            intens_proj_abs = meshes.get_column_flat('intens_proj_abs:{}'.format(dataset), components)
            mus = meshes.get_column_flat('mus', components)
            areas = meshes.get_column_flat('areas_si', components)

            # note that the intensities are already projected but are per unit area
            # so we need to multiply by the /projected/ area of each triangle (thus the extra mu)
            return {'rv': np.average(rvs, weights=intens_proj_abs*areas*mus*visibilities)}

        elif method=='LC':

            visibilities = meshes.get_column_flat('visibilities')

            if np.all(visibilities==0):
                # then no triangles are visible, so we should return nan - probably shouldn't ever happen for lcs
                return {'flux': np.nan}

            intens_proj_rel = meshes.get_column_flat("intens_proj_rel:{}".format(dataset), components)
            mus = meshes.get_column_flat('mus', components)
            areas = meshes.get_column_flat('areas_si', components)

            # intens_proj is the intensity in the direction of the observer per unit surface area of the triangle
            # areas is the area of each triangle
            # areas*mus is the area of each triangle projected in the direction of the observer
            # visibilities is 0 for hidden, 0.5 for partial, 1.0 for visible
            # areas*mus*visibilities is the visibile projected area of each triangle (ie half the area for a partially-visible triangle)
            # so, intens_proj*areas*mus*visibilities is the intensity in the direction of the observer per the observed projected area of that triangle
            # and the sum of these values is the observed flux

            # note that the intensities are already projected but are per unit area
            # so we need to multiply by the /projected/ area of each triangle (thus the extra mu)
            return {'flux': np.sum(intens_proj_rel*areas*mus*visibilities)/(distance**2)+l3}


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

        # Let's create a dictionary to handle how each dataset should scale between
        # absolute and relative intensities.
        self._pblum_scale = {}

        # We'll also keep track of a conservative maximum r (from center of star to triangle, in real units).
        # This will be computed and stored when the periastron mesh is added as a standard
        self._max_r = None

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

        # if mesh_method == 'marching':
        #     maxpoints = int(kwargs.get('maxpoints', self.maxpoints))
        #     if N>=(maxpoints-1):
        #         raise ValueError(("Maximum number of triangles reached ({}). "
        #                           "Consider raising the value of the parameter "
        #                           "'maxpoints', or "
        #                           "decrease the mesh density. It is also "
        #                           "possible that the equipotential surface is "
        #                           "not closed.").format(N))


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
        self.volume_at_periastron = mesh_peri.volume

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

        self.reset_time(time)


        #-- Get current position/euler information
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
            d = self.instantaneous_distance(xs, ys, zs, self.sma)

            # TODO: TESTING should this be unscaled with the new scale or old scale?
            # self._scale = d
            target_volume = self.get_target_volume(ethetas[self.ind_self], scaled=False)
            logger.info("volume conservation: target_volume={}".format(target_volume))


            # TODO: need to send a better guess for Omega0
            Phi = libphoebe.roche_Omega_at_vol(target_volume,
                                               q, F, d,
                                               Omega0=Phi)

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

            # Now we'll save the current mesh in its orbit
            self._mesh = mesh.Mesh.from_scaledproto(scaledprotomesh,
                                                    pos, vel, euler,
                                                    polar_dir*self.freq_rot)

        else:

            # Since we don't have to handle volume conservation, we'll just
            # grab the protomesh standard (unscaled) and scale and place it
            # in orbit in one step
            protomesh = self.get_standard_mesh(scaled=False)
            # TODO: can we avoid an extra copy here?

            # Now we'll save the current mesh in its orbit
            self._mesh = mesh.Mesh.from_proto(protomesh,
                                              self._scale,
                                              pos, vel, euler,
                                              polar_dir*self.freq_rot)


        # Lastly, we'll recompute physical quantities (not observables) if
        # needed for this time-step.
        if not self.mesh.loggs.for_computations or self.needs_recompute_instantaneous:
            self._compute_instantaneous_quantities(xs, ys, zs)

            # Now fill local instantaneous quantities
            self._fill_loggs()
            self._fill_gravs()
            self._fill_teffs()
            self._fill_abuns()

        return


    def compute_luminosity(self, dataset, **kwargs):
        """
        """
        # areas are the NON-projected areas of each surface element.  We'll be
        # integrating over normal intensities, so we don't need to worry about
        # multiplying by mu to get projected areas.
        areas = self.mesh.areas_si

        # intens_norm_abs are directly out of the passbands module and are
        # emergent normal intensities in this dataset's passband/atm in absolute units
        intens_norm_abs = self.mesh['intens_norm_abs:{}'.format(dataset)].centers

        # The luminosity will be the integrated NORMAL intensities, but since some
        # limbdarkening laws still darken/brighten at mu=0, we'll compute the
        # limbdarkening for each element as if it were at mu=0 (directly along
        # the normal).
        ld_coeffs = kwargs.get('ld_coeffs', [0.0,0.0])
        ld_func = kwargs.get('ld_func', 'logarithmic')
        ld = getattr(limbdark, 'ld_{}'.format(ld_func))(np.zeros(len(self.mesh.mus)), ld_coeffs)

        # TODO: remove this once ld is tested
        ld = 1.0

        # Our total integrated intensity in absolute units (luminosity) is now
        # simply the sum of the normal emergent intensities times pi (to account
        # for intensities emitted in all directions across the solid angle),
        # limbdarkened as if they were at mu=0, and multiplied by their respective areas

        total_integrated_intensity = np.sum(intens_norm_abs*ld*areas) * np.pi

        # NOTE: when this is computed the first time (for the sake of determing
        # pblum_scale), get_pblum_scale will return 1.0
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

            self.mesh.update_columns_dict({'{}:{}'.format(key, dataset): col})

        self.populated_at_time.append(dataset)

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

        return new_mesh, sma, mesh_args

    def _fill_teffs(self, **kwargs):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """

        self.mesh.update_columns(teffs=self.teff)

    def _fill_abuns(self, abun=0.0):
        """
        [NOT IMPLEMENTED]

        :raises NotImplementedError: because it isn't
        """
        self.mesh.update_columns(abuns=abun)

    def _populate_ifm(self, dataset, passband, **kwargs):
        """
        [NOT IMPLEMENTED]

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`

        :raises NotImplementedError: because it isn't
        """

        raise NotImplementedError

    def _populate_rv(self, dataset, passband, **kwargs):
        """
        [NOT IMPLEMENTED]


        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`

        :raises NotImplementedError: because it isn't
        """

        raise NotImplementedError


    def _populate_lc(self, dataset, passband, **kwargs):
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
    def __init__(self, F, Phi, masses, sma, ecc, freq_rot, teff, gravb_bol, gravb_law, abun, mesh_method='marching', dynamics_method='keplerian', ind_self=0, ind_sibling=1, comp_no=1, datasets=[], do_rv_grav=False, features=[], **kwargs):
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
        # NOTE: self.q may be flipped her for the secondary
        self.Phi = roche.pot_for_component(Phi, self.q, self.comp_no)

        self.teff = teff
        self.gravb_bol = gravb_bol
        self.gravb_law = gravb_law
        self.abun = abun

        self.features = features

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
        # NOTE: we need F for roche geometry (marching, reprojection), but freq_rot for ctrans.place_in_orbit and rotstar.marching


        masses = [b.get_value('mass', component=star, context='component', unit=u.solMass) for star in starrefs]
        if b.hierarchy.get_parent_of(component) != 'component':
            sma = b.get_value('sma', component=label_orbit, context='component', unit=u.solRad)
            ecc = b.get_value('ecc', component=label_orbit, context='component')
        else:
            # single star case
            # here sma is meaningless, but we'll compute the mesh using the polar radius as the scaling factor
            sma = 1.0 #b.get_value('rpole', component=star, context='component', unit=u.solRad)
            ecc = 0.0

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
            mesh_kwargs['delta'] = b.get_value('delta', component=component, compute=compute, **kwargs) if compute is not None else 0.1
            mesh_kwargs['maxpoints'] = b.get_value('maxpoints', component=component, compute=compute, **kwargs) if compute is not None else 1e5
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
            feature_cls = globals()[feature_ps.method.title()]
            features.append(feature_cls.from_bundle(b, feature))

        return cls(F, Phi, masses, sma, ecc, freq_rot, teff, gravb_bol, gravb_law,
                abun, mesh_method, dynamics_method, ind_self, ind_sibling, comp_no,
                datasets=datasets, do_rv_grav=do_rv_grav, features=features, **mesh_kwargs)

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
        """
        return self.ecc != 0


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
            return self.volume_at_periastron/self._scale**3
        else:
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

        mesh_args = (q, F, d, Phi)

        if mesh_method == 'marching':
            # Phi = kwargs.get('Phi', self.Phi_user)  # NOTE: self.Phi_user is not corrected for the secondary star, but that's fine because we pass primary vs secondary as choice
            # q = 1./self.q if self.comp_no == 2 else self.q  # NOTE: undo the inversion so this is ALWAYS Mp/Ms

            delta = kwargs.get('delta', self.delta)
            maxpoints = int(kwargs.get('maxpoints', self.maxpoints))


            if self.distortion_method == 'roche':
                # TODO: check whether roche or misaligned roche from values of incl, etc!!!!

                # TODO: need to figure this out, this currently causes issues
                # with large sma (too many triangles).  rpole*sma /helps/ but
                # doesn't quite do the right thing.
                rpole = libphoebe.roche_pole(*mesh_args)
                delta *= rpole
                # print delta

                # print mesh_args, delta

                new_mesh = libphoebe.roche_marching_mesh(*mesh_args,
                                                         delta=delta,
                                                         choice=0,
                                                         max_triangles=maxpoints,
                                                         vertices=True,
                                                         triangles=True,
                                                         centers=True,
                                                         vnormals=True,
                                                         tnormals=True,
                                                         cnormals=False,
                                                         vnormgrads=True,
                                                         cnormgrads=False,
                                                         areas=True,
                                                         volume=True)

                # TODO: which volume(s) do we want to report?  Either way, make
                # sure to do the same for the OC case and rotstar
                av = libphoebe.roche_area_volume(*mesh_args,
                                                 choice=0,
                                                 larea=True,
                                                 lvolume=True)

                new_mesh['volume'] = av['lvolume']

                new_mesh['normgrads'] = new_mesh.pop('vnormgrads')
                new_mesh['velocities'] = np.zeros(new_mesh['vertices'].shape)


            elif self.distortion_method == 'rotstar':

                # freq_rot (1./d)
                omega = rotstar.rotfreq_to_omega(self.freq_rot, scale=sma, solar_units=True)


                sma /= 695700000.0

                # rpole = libphoebe.rotstar_pole(*mesh_args) / sma # / 695700000.0  # TODO: is the 6.957 necessary?
                rpole = rotstar.potential2rpole(Phi, self.freq_rot, solar_units=True)
                # rpole is now in solar units
                # rpole /= 695700000.0
                rpole /= sma
                delta *= rpole



                mesh_args = (omega, Phi)

                print "*** rotstar_marching_mesh", mesh_args, self.freq_rot, sma, rpole, delta
                delta = 0.01

                new_mesh = libphoebe.rotstar_marching_mesh(*mesh_args,
                                               delta=delta,
                                               max_triangles=maxpoints,
                                               vertices=True,
                                               triangles=True,
                                               centers=True,
                                               vnormals=True,
                                               tnormals=True,
                                               cnormals=False,
                                               vnormgrads=True,
                                               cnormgrads=False,
                                               areas=True,
                                               volume=True)

                # TODO: enable this once its supported by libphoebe
                # av = libphoebe.rotstar_area_volume(*mesh_args,
                #                                    larea=True,
                #                                    lvolume=True)

                # new_mesh['volume'] = av['lvolume']

                new_mesh['normgrads'] = new_mesh.pop('vnormgrads')
                new_mesh['velocities'] = np.zeros(new_mesh['vertices'].shape)

            elif self.distortion_method == 'sphere':
                # TODO: implement this (discretize and save mesh_args)
                raise NotImplementedError("'sphere' distortion method not yet supported - try roche")
            elif self.distortion_method == 'nbody':
                # TODO: implement this (discretize and save mesh_args)
                raise NotImplementedError("'nbody' distortion method not yet supported - try roche")
            else:
                raise NotImplementedError

        elif mesh_method == 'wd':

            N = int(kwargs.get('gridsize', self.gridsize))

            the_grid = potentials.discretize_wd_style(N, *mesh_args)
            new_mesh = mesh.wd_grid_to_mesh_dict(the_grid, q, F, d)

        else:
            raise NotImplementedError("mesh method '{}' is not supported".format(mesh_method))

        return new_mesh, sma, mesh_args

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

    def _fill_loggs(self, mesh=None):
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


    def _fill_teffs(self, mesh=None, **kwargs):
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

        if self.gravb_law == 'espinosa':
            # TODO: check whether we want the automatically inverted q or not
            q = self.q  # NOTE: this is automatically flipped to be 1./q for secondary components
            F = self.syncpar
            sma = self.sma

            # TODO NOW: rewrite this to work in unscaled units

            # To compute the filling factor, we're gonna cheat a little bit: we
            # should compute the ratio of the tip radius to the first Lagrangian
            # point. However, L1 might be poorly defined for weird geometries
            # so we approximate it as 1.5 times the polar radius.
            # TODO NOW: rp doesn't seem to be used anywhere...
            rp = self._instantaneous_rpole  # should be in Rsol

            # TODO NOW: is this supposed to be the scaled or unscaled rs???
            maxr = self.get_standard_mesh(scaled=True).rs.for_computations.max()

            L1 = roche.exact_lagrangian_points(q, F=F, d=1.0, sma=sma)[0]
            rho = maxr / L1

            gravb = roche.zeipel_gravb_binary()(np.log10(q), rho)[0][0]

            logger.info("gravb(Espinosa): F = {}, q = {}, filling factor = {} --> gravb = {}".format(F, q, rho, gravb))
            if gravb>1.0 or gravb<0:
                raise ValueError('Invalid gravity darkening parameter beta={}'.format(gravb))

        elif self.gravb_law == 'claret':
            logteff = np.log10(self.teff)
            logg = np.log10(self._instantaneous_gpole)
            abun = self.abun
            axv, pix = roche.claret_gravb()
            gravb = interp_nDgrid.interpolate([[logteff], [logg], [abun]], axv, pix)[0][0]

            logger.info('gravb(Claret): teff = {:.3f}, logg = {:.6f}, abun = {:.3f} ---> gravb = {:.3f}'.format(10**logteff, logg, abun, gravb))

        # TODO: ditch support for polar teff as input param

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
        if typ == 'mean':
            # TODO NOW: can this be done on an unscaled mesh? (ie can we fill teffs in the protomesh or do areas need to be scaled to real units)
            # Convert from mean to polar by dividing total flux by gravity darkened flux (Ls drop out)
            Tpole = Teff*(np.sum(mesh.areas) / np.sum(mesh.gravs.centers*mesh.areas))**(0.25)
        elif typ == 'polar':
            Tpole = Teff
        else:
            raise ValueError("Cannot interpret temperature type '{}' (needs to be one of ['mean','polar'])".format(typ))

        self._instantaneous_teffpole = Tpole

        # Now we can compute the local temperatures.
        teffs = (mesh.gravs.for_computations * Tpole**4)**0.25

        for spot in self.spots:
            teffs = spot.process_teffs(teffs, mesh.coords_for_computations)

        mesh.update_columns(teffs=teffs)

        # logger.info("derived effective temperature (Zeipel) (%.3f <= teff <= %.3f, Tp=%.3f)"%(teffs.min(), teffs.max(), Tpole))

    def _fill_abuns(self, mesh=None, abun=0.0):
        """
        TODO: add documentation
        """
        if mesh is None:
            mesh = self.mesh

        # TODO: support from frontend

        mesh.update_columns(abuns=abun)


    def _populate_ifm(self, dataset, passband, **kwargs):
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
        ld_coeffs = kwargs.get('ld_coeffs', [0.5,0.5])
        ld_func = kwargs.get('ld_func', 'logarithmic')
        atm = kwargs.get('atm', 'kurucz')
        boosting_alg = kwargs.get('boosting_alg', 'none')

        lc_cols = self._populate_lc(dataset, **kwargs)

        # LC cols should do for interferometry - at least for continuum
        cols = lc_cols

        return cols

    def _populate_rv(self, dataset, passband, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`
        """

        # print "*** Star._populate_rv"
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

        rvs = -1*self.mesh.velocities.for_computations[:,2]

        # Gravitational redshift
        if self.do_rv_grav:
            rv_grav = c.G*(self.mass*u.solMass)/(self._instantaneous_rpole*u.solRad)/c.c
            # rvs are in solrad/d internally
            rv_grav = rv_grav.to('solRad/d').value

            rvs += rv_grav

        cols = lc_cols
        cols['rv'] = rvs
        return cols


    def _populate_lc(self, dataset, passband, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`

        :raises NotImplementedError: if lc_method is not supported
        """

        lc_method = kwargs.get('lc_method', 'numerical')  # TODO: make sure this is actually passed

        ld_coeffs = kwargs.get('ld_coeffs', [0.5,0.5])
        ld_func = kwargs.get('ld_func', 'logarithmic')
        atm = kwargs.get('atm', 'blackbody')
        boosting_alg = kwargs.get('boosting_alg', 'none')

        pblum = kwargs.get('pblum', 4*np.pi)


        if lc_method=='numerical':

            if passband not in self._pbs.keys():
                passband_fname = passbands._pbtable[passband]['fname']
                logger.info("using ptf file: {}".format(passband_fname))
                pb = passbands.Passband.load(passband_fname)

                self._pbs[passband] = pb

            # intens_norm_abs are the normal emergent passband intensities:
            intens_norm_abs = self._pbs[passband].Inorm(Teff=self.mesh.teffs.for_computations,
                                                        logg=self.mesh.loggs.for_computations,
                                                        met=self.mesh.abuns.for_computations,
                                                        atm=atm)

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
            # TODO: should we mutliply velocities by -1 (z convention)?
            ampl_boost = 1.0 + alpha_b * self.mesh.velocities.for_computations[:,2]/37241.94167601236


            # Limb-darkening

            # TODO: the abs shouldn't be necessary here, but for some reason
            # is causing issues when ld_logarithmic with negative mus (well
            # I mean the reason it fails is obvious)
            ld = getattr(limbdark, 'ld_{}'.format(ld_func))(np.abs(self.mesh.mus_for_computations), ld_coeffs)

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
        # TODO: should F be fixed at 1 - is this the job of the frontend or backend?

        Phi = kwargs.get('Phi', self.Phi)  # NOTE: self.Phi automatically corrects for the secondary star
        q = self.q  # NOTE: this is automatically flipped to be 1./q for secondary components

        mesh_args = (q, F, d, Phi)

        if mesh_method == 'marching':
            # Phi = kwargs.get('Phi', self.Phi_user)  # NOTE: self.Phi_user is not corrected for the secondary star, but that's fine because we pass primary vs secondary as choice
            # q = 1./self.q if self.comp_no == 2 else self.q  # NOTE: undo the inversion so this is ALWAYS Mp/Ms

            delta = kwargs.get('delta', self.delta)
            maxpoints = int(kwargs.get('maxpoints', self.maxpoints))


            if self.distortion_method == 'roche':
                # TODO: check whether roche or misaligned roche from values of incl, etc!!!!

                rpole = libphoebe.roche_pole(*mesh_args)
                delta *= rpole

                new_mesh = libphoebe.roche_marching_mesh(*mesh_args,
                                                         choice=0,
                                                         delta=delta,
                                                         max_triangles=maxpoints,
                                                         vertices=True,
                                                         triangles=True,
                                                         centers=True,
                                                         vnormals=True,
                                                         tnormals=True,
                                                         cnormals=False,
                                                         vnormgrads=True,
                                                         cnormgrads=False,
                                                         areas=True,
                                                         volume=True)

                new_mesh['normgrads'] = new_mesh.pop('vnormgrads')
                new_mesh['velocities'] = np.zeros(new_mesh['vertices'].shape)

            elif self.distortion_method == 'nbody':
                # TODO: implement this? - can OCs be done in NBody mode?
                raise NotImplementedError("'nbody' distortion method not yet supported - try roche")
            else:
                raise NotImplementedError

        elif mesh_method == 'wd':
            # TODO: WD-style meshing for OCs (Angela)
            raise NotImplementedError("WD-meshing not yet supported for overcontacts")

        else:
            raise NotImplementedError("mesh method '{}' is not supported".format(mesh_method))

        return new_mesh, sma, mesh_args


    def _compute_instantaneous_quantities(self, xs, ys, zs, **kwargs):
        """
        TODO: add documentation
        """
        pass  # TODO: do we need any of these things for overcontacts?

        #q = self.q
        #d = kwargs.get('d') if 'd' in kwargs.keys() else self.instantaneous_distance(xs, ys, zs, self.sma)

        #r_pole_ = potentials.project_onto_potential(np.array((0, 0, 1e-5)), *self._mesh_args).r # instantaneous unitless r_pole (not rpole defined at periastron)
        #g_pole = np.sqrt(dBinaryRochedx(r_pole_, d, q, self.F)**2 + dBinaryRochedz(r_pole_, d, q, self.F)**2)
        #rel_to_abs = c.G.si.value*c.M_sun.si.value*self.masses[self.ind_self]/(self.sma*c.R_sun.si.value)**2*100. # 100 for m/s**2 -> cm/s**2

        #self._instantaneous_gpole = g_pole * rel_to_abs
        #self._instantaneous_rpole = np.sqrt((r_pole_*r_pole_).sum())

    def _fill_loggs(self):
        """
        TODO: add documentation

        Calculate local surface gravity

        GMSunNom = 1.3271244e20 m**3 s**-2
        RSunNom = 6.597e8 m
        """
        self.mesh.update_columns(loggs=0.0)


    def _fill_gravs(self, **kwargs):
        """
        TODO: add documentation

        requires _fill_loggs to have been called
        """

        self.mesh.update_columns(gravs=0.0)

    def _fill_teffs(self, **kwargs):
        """
        [NOT IMPLEMENTED]
        requires _fill_loggs and _fill_gravs to have been called
        """
        self.mesh.update_columns(teffs=0.0)


    def _fill_abuns(self, abun=0.0):
        """
        TODO: add documentation
        """
        self.mesh.update_columns(abuns=abun)

    def _populate_ifm(self, dataset, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`
        """
        raise NotImplementedError

    def _populate_rv(self, dataset, passband, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`
        """

        # print "*** Star._populate_rv"
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
            rv_grav = c.G*(self.mass*u.solMass)/(self._instantaneous_rpole*u.solRad)/c.c
            # rvs are in solrad/d internally
            rv_grav = rv_grav.to('solRad/d').value

            rvs += rv_grav

        cols = lc_cols
        cols['rv'] = rvs
        return cols


    def _populate_lc(self, dataset, passband, **kwargs):
        """
        TODO: add documentation

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`

        :raises NotImplementedError: if lc_method is not supported
        """

        lc_method = kwargs.get('lc_method', 'numerical')  # TODO: make sure this is actually passed

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









class Feature(object):
    def __init__(self, *args, **kwargs):
        pass

class Spot(Feature):
    def __init__(self, colat, colon, radius, relteff, **kwargs):
        super(Spot, self).__init__(**kwargs)
        self._colat = colat
        self._colon = colon
        self._radius = radius
        self._relteff = relteff

        x = np.sin(colat)*np.cos(colon)
        y = np.sin(colat)*np.sin(colon)
        z = np.cos(colat)
        self._pointing_vector = np.array([x,y,z])

    @classmethod
    def from_bundle(cls, b, feature):
        """
        """

        feature_ps = b.get_feature(feature)
        colat = feature_ps.get_value('colat', unit=u.rad)
        colon = feature_ps.get_value('colon', unit=u.rad)
        radius = feature_ps.get_value('radius', unit=u.rad)
        relteff = feature_ps.get_value('relteff', unit=u.dimensionless_unscaled)
        return cls(colat, colon, radius, relteff)

    def process_teffs(self, teffs, coords):
        """
        """

        cos_alpha_coords = np.dot(coords, self._pointing_vector) / np.linalg.norm(coords, axis=1)
        cos_alpha_spot = np.cos(self._radius)

        filter = cos_alpha_coords > cos_alpha_spot

        teffs[filter] = teffs[filter] * self._relteff

        return teffs
