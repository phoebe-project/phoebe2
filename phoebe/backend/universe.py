import numpy as np
from scipy.optimize import newton
from scipy.special import sph_harm as Y
from math import sqrt, sin, cos, acos, atan2, trunc, pi
import sys, os
import copy

from phoebe.atmospheres import passbands
from phoebe.distortions import roche, rotstar
from phoebe.backend import eclipse, oc_geometry, mesh, mesh_wd
from phoebe.utils import _bytes
import libphoebe

from phoebe import u
from phoebe import c
from phoebe import conf

import logging
logger = logging.getLogger("UNIVERSE")
logger.addHandler(logging.NullHandler())

_basedir = os.path.dirname(os.path.abspath(__file__))
_pbdir = os.path.abspath(os.path.join(_basedir, '..', 'atmospheres', 'tables', 'passbands'))
_skip_filter_checks = {'check_default': False, 'check_visible': False}

"""
Class/SubClass Structure of Universe.py:

System - container for all Bodies

Body - general Class for all Bodies
    any new type of object needs to subclass Body and override the following:
        * is_convex
        * needs_remesh
        * needs_recompute_instantaneous
        * _build_mesh
        * _populate_lc
        * _populate_rv
+ Star(Body) - subclass of Body that acts as a general class for any type of deformed star defined by requiv
    any new type of Star needs to subclass Star and override the following:
        * _rpole_func
        * _gradOmega_func
        * instantaneous_mesh_args
        * _build_mesh
  + Star_roche(Star) [not allowed as single star]
  + Star_roche_envelope_half(Star)
  + Star_rotstar(Star)
  + Star_sphere(Star)
+ Envelope(Body) - subclass of Body that contains two Star_roche_envelope_half instances

If creating a new subclass of Body, make sure to add it to top-level
_get_classname function if the class is not simply the title-case of the
component kind in the Bundle

Feature - general Class of all features: any new type of feature needs to subclass Feature
+ Spot(Feature)
+ Pulsation(Feature)

"""

def g_rel_to_abs(mass, sma):
    return c.G.si.value*c.M_sun.si.value*mass/(sma*c.R_sun.si.value)**2*100. # 100 for m/s**2 -> cm/s**2

def _get_classname(kind, distortion_method):
    kind = kind.title()
    if kind == 'Envelope':
        return 'Envelope'
    elif kind == 'Star':
        # Star_roche, Star_rotstar, Star_sphere
        # NOTE: Star_roche_envelope_half should never be called directly, but
        # rather through the Envelope class above.
        return 'Star_{}'.format(distortion_method)
    else:
        return kind

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

def _estimate_delta(ntriangles, area):
    """
    estimate the value for delta to send to marching based on the number of
    requested triangles and the expected surface area of mesh
    """
    return np.sqrt(4./np.sqrt(3) * float(area) / float(ntriangles))


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

        self.is_first_refl_iteration = True

        for body in self._bodies.values():
            body.system = self
            body.dynamics_method = dynamics_method
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
                compute_ps = b.get_compute(compute=compute, **_skip_filter_checks)
            else:
                # then hopefully compute is the parameterset
                compute_ps = compute

            eclipse_method = compute_ps.get_value(qualifier='eclipse_method', eclipse_method=kwargs.get('eclipse_method', None), **_skip_filter_checks)
            horizon_method = compute_ps.get_value(qualifier='horizon_method', horizon_method=kwargs.get('horizon_method', None), **_skip_filter_checks)
            dynamics_method = compute_ps.get_value(qualifier='dynamics_method', dynamics_method=kwargs.get('dynamics_method', None), **_skip_filter_checks)
            irrad_method = compute_ps.get_value(qualifier='irrad_method', irrad_method=kwargs.get('irrad_method', None), **_skip_filter_checks)
            boosting_method = compute_ps.get_value(qualifier='boosting_method', boosting_method=kwargs.get('boosting_method', None), **_skip_filter_checks)
        else:
            eclipse_method = 'native'
            horizon_method = 'boolean'
            dynamics_method = 'keplerian'
            irrad_method = 'none'
            boosting_method = 'none'
            compute_ps = None

        # NOTE: here we use globals()[Classname] because getattr doesn't work in
        # the current module - now this doesn't really make sense since we only
        # support stars, but eventually the classname could be Disk, Spot, etc
        if 'dynamics_method' in kwargs.keys():
            # already set as default above
            _dump = kwargs.pop('dynamics_method')

        meshables = hier.get_meshables()
        def get_distortion_method(hier, compute_ps, component, **kwargs):
            if hier.get_kind_of(component) in ['envelope']:
                return 'roche'

            if compute_ps is None:
                return 'roche'

            if compute_ps.get_value(qualifier='mesh_method', component=component, mesh_method=kwargs.get('mesh_method', None), **_skip_filter_checks)=='wd':
                return 'roche'

            return compute_ps.get_value(qualifier='distortion_method', component=component, distortion_method=kwargs.get('distortion_method', None), **_skip_filter_checks)

        bodies_dict = {comp: globals()[_get_classname(hier.get_kind_of(comp), get_distortion_method(hier, compute_ps, comp, **kwargs))].from_bundle(b, comp, compute, dynamics_method=dynamics_method, datasets=datasets, **kwargs) for comp in meshables}

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
        return list(self._bodies.keys())

    def values(self):
        """
        TODO: add documentation
        """
        return list(self._bodies.values())

    @property
    def bodies(self):
        """
        TODO: add documentation
        """
        return list(self.values())

    def get_body(self, component):
        """
        TODO: add documentation
        """
        if component in self._bodies.keys():
            return self._bodies[component]
        else:
            # then hopefully we're a child star of an contact_binary envelope
            parent_component = self._parent_envelope_of[component]
            return self._bodies[parent_component].get_half(component)

    @property
    def needs_recompute_instantaneous(self):
        return np.any([b.needs_recompute_instantaneous for b in self.bodies])

    @property
    def mesh_bodies(self):
        """
        """
        bodies = []
        for body in self.bodies:
            if isinstance(body, Envelope):
                bodies += body._halves
            else:
                bodies += [body]

        return bodies

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

        # return mesh.Meshes({c:b for c,b in self._bodies.items() if b is not None}, self._parent_envelope_of)
        return mesh.Meshes(self._bodies, self._parent_envelope_of)

    def reset(self, force_remesh=False, force_recompute_instantaneous=False):
        self.is_first_refl_iteration = True
        for body in self.bodies:
            body.reset(force_remesh=force_remesh, force_recompute_instantaneous=force_recompute_instantaneous)


    def update_positions(self, time, xs, ys, zs, vxs, vys, vzs,
                         ethetas, elongans, eincls,
                         ds=None, Fs=None, ignore_effects=False):
        """
        TODO: add documentation

        all arrays should be for the current time, but iterable over all bodies
        """
        logger.debug('system.update_positions ignore_effects={}'.format(ignore_effects))
        self.xs = np.array(_value(xs))
        self.ys = np.array(_value(ys))
        self.zs = np.array(_value(zs))

        for starref,body in self.items():
            body.update_position(time, xs, ys, zs, vxs, vys, vzs,
                                 ethetas, elongans, eincls,
                                 ds=ds, Fs=Fs, ignore_effects=ignore_effects)


    def populate_observables(self, time, kinds, datasets, ignore_effects=False):
        """
        TODO: add documentation

        ignore_effects: whether to ignore reflection and features (useful for computing luminosities)
        """


        if self.irrad_method != 'none' and not ignore_effects:
            # TODO: only for kinds that require intensities (i.e. not orbit or
            # dynamical RVs, etc)
            self.handle_reflection()

        for kind, dataset in zip(kinds, datasets):
            for starref, body in self.items():
                body.populate_observable(time, kind, dataset, ignore_effects=ignore_effects)

    def handle_reflection(self,  **kwargs):
        """
        """
        if self.irrad_method == 'none':
            return

        if not self.needs_recompute_instantaneous and not self.is_first_refl_iteration:
            logger.debug("reflection: using teffs from previous iteration")
            return

        if 'wd' in [body.mesh_method for body in self.bodies]:
            raise NotImplementedError("reflection not supported for WD-style meshing")

        # meshes is an object which allows us to easily access and update columns
        # in the meshes *in memory*.  That is meshes.update_columns will propagate
        # back to the current mesh for each body.
        meshes = self.meshes

        # reflection needs bolometric, energy weighted, normal intensities.
        logger.debug("reflection: computing bolometric intensities")
        fluxes_intrins_per_body = []
        for starref, body in self.items():
            if body.mesh is None: continue
            abs_normal_intensities = passbands.Inorm_bol_bb(Teff=body.mesh.teffs.for_computations,
                                                            atm='blackbody',
                                                            photon_weighted=False)

            fluxes_intrins_per_body.append(abs_normal_intensities * np.pi)

        fluxes_intrins_flat = meshes.pack_column_flat(fluxes_intrins_per_body)

        if len(fluxes_intrins_per_body) == 1 and np.all([body.is_convex for body in self.bodies]):
            logger.info("skipping reflection because only 1 (convex) body")
            return

        elif np.all([body.is_convex for body in self.bodies]):
            logger.debug("handling reflection (convex case), method='{}'".format(self.irrad_method))

            vertices_per_body = list(meshes.get_column('vertices').values())
            triangles_per_body = list(meshes.get_column('triangles').values())
            normals_per_body = list(meshes.get_column('vnormals').values())
            areas_per_body = list(meshes.get_column('areas').values())
            irrad_frac_refl_per_body = list(meshes.get_column('irrad_frac_refl', computed_type='for_computations').values())
            teffs_intrins_per_body = list(meshes.get_column('teffs', computed_type='for_computations').values())

            ld_func_and_coeffs = [tuple([_bytes(body.ld_func['bol'])] + [np.asarray(body.ld_coeffs['bol'])]) for body in self.bodies]
            logger.debug("irradiation ld_func_and_coeffs: {}".format(ld_func_and_coeffs))
            fluxes_intrins_and_refl_per_body = libphoebe.mesh_radiosity_problem_nbody_convex(vertices_per_body,
                                                                                       triangles_per_body,
                                                                                       normals_per_body,
                                                                                       areas_per_body,
                                                                                       irrad_frac_refl_per_body,
                                                                                       fluxes_intrins_per_body,
                                                                                       ld_func_and_coeffs,
                                                                                       _bytes(self.irrad_method.title()),
                                                                                       support=_bytes('vertices')
                                                                                       )

            fluxes_intrins_and_refl_flat = meshes.pack_column_flat(fluxes_intrins_and_refl_per_body)

        else:
            logger.debug("handling reflection (general case), method='{}'".format(self.irrad_method))

            vertices_flat = meshes.get_column_flat('vertices') # np.ndarray
            triangles_flat = meshes.get_column_flat('triangles') # np.ndarray
            normals_flat = meshes.get_column_flat('vnormals') # np.ndarray
            areas_flat = meshes.get_column_flat('areas') # np.ndarray
            irrad_frac_refl_flat = meshes.get_column_flat('irrad_frac_refl', computed_type='for_computations') # np.ndarray

            ld_func_and_coeffs = [tuple([_bytes(body.ld_func['bol'])] + [np.asarray(body.ld_coeffs['bol'])]) for body in self.mesh_bodies] # list
            ld_inds_flat = meshes.pack_column_flat({body.comp_no: np.full(fluxes.shape, body.comp_no-1) for body, fluxes in zip(self.mesh_bodies, fluxes_intrins_per_body)}) # np.ndarray

            fluxes_intrins_and_refl_flat = libphoebe.mesh_radiosity_problem(vertices_flat,
                                                                            triangles_flat,
                                                                            normals_flat,
                                                                            areas_flat,
                                                                            irrad_frac_refl_flat,
                                                                            fluxes_intrins_flat,
                                                                            ld_func_and_coeffs,
                                                                            ld_inds_flat,
                                                                            _bytes(self.irrad_method.title()),
                                                                            support=_bytes('vertices')
                                                                            )



        teffs_intrins_flat = meshes.get_column_flat('teffs', computed_type='for_computations')

        # update the effective temperatures to give this same bolometric
        # flux under stefan-boltzmann. These effective temperatures will
        # then be used for all passband intensities.
        teffs_intrins_and_refl_flat = teffs_intrins_flat * (fluxes_intrins_and_refl_flat / fluxes_intrins_flat) ** (1./4)

        nanmask = np.isnan(teffs_intrins_and_refl_flat)
        if np.any(nanmask):
            raise ValueError("irradiation resulted in nans for teffs")

        meshes.set_column_flat('teffs', teffs_intrins_and_refl_flat)

        if not self.needs_recompute_instantaneous:
            logger.debug("reflection: copying updated teffs to standard mesh")
            theta = 0.0
            standard_meshes = mesh.Meshes({body.component: body._standard_meshes[theta] for body in self.mesh_bodies})
            standard_meshes.set_column_flat('teffs', teffs_intrins_and_refl_flat)

            self.is_first_refl_iteration = False

    def handle_eclipses(self, expose_horizon=False, **kwargs):
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
            logger.debug("system.handle_eclipses: determining if eclipses are possible from instantaneous_maxr")
            max_rs = [body.instantaneous_maxr for body in self.bodies]
            # logger.debug("system.handle_eclipses: max_rs={}".format(max_rs))
            for i in range(0, len(max_rs)-1):
                for j in range(i+1, len(max_rs)):
                    proj_sep_sq = sum([(c[i]-c[j])**2 for c in (self.xs,self.ys)])
                    max_sep_ecl = max_rs[i] + max_rs[j]

                    if proj_sep_sq < (1.05*max_sep_ecl)**2:
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

        logger.debug("system.handle_eclipses: possible_eclipse={}, expose_horizon={}, calling {} with kwargs {}".format(possible_eclipse, expose_horizon, eclipse_method, ecl_kwargs))

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

        # weights is also a dictionary with keys being the component labels
        # and values and np array of weights.
        if weights is not None:
            meshes.update_columns('weights', weights)

        return horizon


    def observe(self, dataset, kind, components=None, **kwargs):
        """
        TODO: add documentation

        Integrate over visible surface elements and return a dictionary of observable values
        """

        meshes = self.meshes
        if kind=='lp':
            def sv(p, p0, w):
                # Subsidiary variable:
                return (p0-p)/(w/2)

            def lorentzian(sv):
                return 1-1./(1+sv**2)

            def gaussian(sv):
                return 1-np.exp(-np.log(2)*sv**2)

            profile_func = kwargs.get('profile_func')
            profile_rest = kwargs.get('profile_rest')
            profile_sv = kwargs.get('profile_sv')
            wavelengths = kwargs.get('wavelengths')
            if profile_func == 'gaussian':
                func = gaussian
            elif profile_func == 'lorentzian':
                func = lorentzian
            else:
                raise NotImplementedError("profile_func='{}' not supported".format(profile_func))

            # By design, the wavelengths array needs to contain central wavelength.
            # For that reason we use an internal wavelengths array which we then
            # interpolate to the requested wavelengths array.
            wmin, wmax = wavelengths.min(), wavelengths.max()
            min_dispersion = (wavelengths[1:]-wavelengths[:-1]).min()
            lower_range = np.arange(profile_rest, wmin-min_dispersion, -min_dispersion)[::-1]
            upper_range = np.arange(profile_rest+min_dispersion, wmax+min_dispersion, min_dispersion)
            internal_wavelengths = np.concatenate((lower_range, upper_range))

            visibilities = meshes.get_column_flat('visibilities', components)

            abs_intensities = meshes.get_column_flat('abs_intensities:{}'.format(dataset), components)
            # mus here will be from the tnormals of the triangle and will not
            # be weighted by the visibility of the triangle
            mus = meshes.get_column_flat('mus', components)
            areas = meshes.get_column_flat('areas_si', components)

            rvs = (meshes.get_column_flat("rvs:{}".format(dataset), components)*u.solRad/u.d).to(u.m/u.s).value
            dls = profile_rest*rvs/c.c.si.value

            line = func(sv(internal_wavelengths, profile_rest, profile_sv))
            lines = np.array([np.interp(wavelengths, internal_wavelengths+dl, line) for dl in dls])
            if not np.any(visibilities):
                avg_line = np.full_like(wavelengths, np.nan)
            else:
                avg_line = np.average(lines, axis=0, weights=abs_intensities*areas*mus*visibilities)

            return {'flux_densities': avg_line}


        elif kind=='rv':
            visibilities = meshes.get_column_flat('visibilities', components)

            if np.all(visibilities==0):
                # then no triangles are visible, so we should return nan
                return {'rv': np.nan}

            rvs = meshes.get_column_flat("rvs:{}".format(dataset), components)
            abs_intensities = meshes.get_column_flat('abs_intensities:{}'.format(dataset), components)
            # mus here will be from the tnormals of the triangle and will not
            # be weighted by the visibility of the triangle
            mus = meshes.get_column_flat('mus', components)
            areas = meshes.get_column_flat('areas_si', components)
            # NOTE: don't need ptfarea because its a float (same for all
            # elements, regardless of component)

            # NOTE: the intensities are already projected but are per unit area
            # so we need to multiply by the /projected/ area of each triangle (thus the extra mu)
            return {'rv': np.average(rvs, weights=abs_intensities*areas*mus*visibilities)}

        elif kind=='lc':
            visibilities = meshes.get_column_flat('visibilities')

            if np.all(visibilities==0):
                # then no triangles are visible, so we should return nan -
                # probably shouldn't ever happen for lcs
                return {'flux': np.nan}

            intensities = meshes.get_column_flat("intensities:{}".format(dataset), components)
            mus = meshes.get_column_flat('mus', components)
            areas = meshes.get_column_flat('areas_si', components)

            # assume that all bodies are using the same passband and therefore
            # will have the same ptfarea.  If this assumption is ever a problem -
            # then we will need to build a flat column based on the component
            # of each element so that ptfarea is an array with the same shape
            # as those above
            for body in self.bodies:
                if body.mesh is not None:
                    if isinstance(body, Envelope):
                        # for envelopes, we'll make the same assumption and just grab
                        # that value stored in the first "half"
                        ptfarea = body._halves[0].get_ptfarea(dataset)
                    else:
                        ptfarea = body.get_ptfarea(dataset)
                    break

            # intensities (Imu) is the intensity in the direction of the observer per unit surface area of the triangle, scaled according to pblum scaling
            # areas is the area of each triangle (using areas_si from the mesh to force SI units)
            # areas*mus is the area of each triangle projected in the direction of the observer
            # visibilities is 0 for hidden, 0.5 for partial, 1.0 for visible
            # areas*mus*visibilities is the visibile projected area of each triangle (ie half the area for a partially-visible triangle)
            # so, intensities*areas*mus*visibilities is the intensity in the direction of the observer per the observed projected area of that triangle
            # and the sum of these values is the observed flux

            # note that the intensities are already projected (Imu) but are per unit area
            # so we need to multiply by the /projected/ area of each triangle (thus the extra mu)

            return {'flux': np.sum(intensities*areas*mus*visibilities)*ptfarea}

        else:
            raise NotImplementedError("observe for dataset with kind '{}' not implemented".format(kind))




class Body(object):
    """
    Body is the base Class for all "bodies" of the System.

    """
    def __init__(self, component, comp_no, ind_self, ind_sibling, masses,
                 ecc, incl, long_an, t0,
                 do_mesh_offset=True,
                 mesh_init_phi=0.0):
        """
        TODO: add documentation
        """

        # TODO: eventually some of this stuff that assumes a BINARY orbit may need to be moved into
        # some subclass of Body (maybe BinaryBody).  These will want to be shared by Star and CustomBody,
        # but probably won't be shared by disk/ring-type objects

        # Let's remember the component number of this star in the parent orbit
        # 1 = primary
        # 2 = secondary
        self.comp_no = comp_no
        self.component = component

        # We need to remember what index in all incoming position/velocity/euler
        # arrays correspond to both ourself and our sibling
        self.ind_self = ind_self
        self.ind_self_vel = ind_self
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
        self.t0 = t0   # t0@system
        self.time = None
        self.inst_vals = {}
        self.true_anom = 0.0
        self.elongan = long_an
        self.eincl = incl
        self.populated_at_time = []

        self.incl_orbit = incl
        self.longan_orbit = long_an

        # Let's create a dictionary to store "standard" protomeshes at different "phases"
        # For example, we may want to store the mesh at periastron and use that as a standard
        # for reprojection for volume conservation in eccentric orbits.
        # Storing meshes should only be done through self.save_as_standard_mesh(theta)
        self._standard_meshes = {}

        self.mesh_init_phi = mesh_init_phi
        self.do_mesh_offset = do_mesh_offset

        # TODO: allow custom meshes (see alpha:universe.Body.__init__)

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
        return False

    @property
    def needs_recompute_instantaneous(self):
        """
        whether the Body needs local quantities recomputed at each time, even
        if needs_remesh == False (instantaneous local quantities will be recomputed
        if needs_remesh=True, whether or not this is True)

        this should be overridden by any subclass of Body
        """
        return True

    @property
    def needs_remesh(self):
        """
        whether the Body needs to be re-meshed (for any reason)

        this should be overridden by any subclass of Body
        """
        return True

    @property
    def instantaneous_maxr(self):
        """
        Recall the maximum r (triangle furthest from the center of the star) of
        this star at the given time

        :return: maximum r
        :rtype: float
        """
        logger.debug("{}.instantaneous_maxr".format(self.component))

        if 'maxr' not in self.inst_vals.keys():
            logger.debug("{}.instantaneous_maxr COMPUTING".format(self.component))

            self.inst_vals['maxr'] = max(self.mesh.rs.centers*self._scale)

        return self.inst_vals['maxr']

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
            return np.average([_value(coords_array[i]) for i in index],
                              weights=[self._get_mass_by_index(i) for i in index])
        else:
            return coords_array[index]

    def _offset_mesh(self, new_mesh):
        if self.do_mesh_offset and self.mesh_method=='marching':
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
            mo = libphoebe.mesh_offseting(new_mesh['area'],
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

        return new_mesh

    def save_as_standard_mesh(self, protomesh):
        """
        TODO: add documentation
        """
        # TODO: allow this to take theta or separation
        theta=0.0

        self._standard_meshes[theta] = protomesh.copy()

        # if theta==0.0:
            # then this is when the object could be most inflated, so let's
            # store the maximum distance to a triangle.  This is then used to
            # conservatively and efficiently estimate whether an eclipse is
            # possible at any given combination of positions
            # mesh = self.get_standard_mesh(theta=0.0, scaled=True)

            # self._max_r = np.sqrt(max([x**2+y**2+z**2 for x,y,z in mesh.centers]))

    def has_standard_mesh(self):
        """
        whether a standard mesh is available
        """
        # TODO: allow this to take etheta and look to see if we have an existing
        # standard close enough
        theta = 0.0
        return theta in self._standard_meshes.keys()

    def get_standard_mesh(self, scaled=True):
        """
        TODO: add documentation
        """
        # TODO: allow this to take etheta and retreive a mesh at that true anomaly
        theta = 0.0
        protomesh = self._standard_meshes[theta] #.copy() # if theta in self._standard_meshes.keys() else self.mesh.copy()

        if scaled:
            # TODO: be careful about self._scale... we may want self._instantaneous_scale
            return mesh.ScaledProtoMesh.from_proto(protomesh, self._scale)
        else:
            return protomesh.copy()

        # return mesh

    def reset(self, force_remesh=False, force_recompute_instantaneous=False):
        if force_remesh:
            logger.debug("{}.reset: forcing remesh and recompute_instantaneous for next iteration".format(self.component))
        elif force_recompute_instantaneous:
            logger.debug("{}.reset: forcing recompute_instantaneous for next iteration".format(self.component))

        if self.needs_remesh or force_remesh:
            self._mesh = None
            self._standard_meshes = {}

        if self.needs_recompute_instantaneous or self.needs_remesh or force_remesh or force_recompute_instantaneous:
            self.inst_vals = {}
            self._force_recompute_instantaneous_next_update_position = True

    def reset_time(self, time, true_anom, elongan, eincl):
        """
        TODO: add documentation
        """
        self.true_anom = true_anom
        self.elongan = elongan
        self.eincl = eincl
        self.time = time
        self.populated_at_time = []

        self.reset()

        return

    def _build_mesh(self, *args, **kwargs):
        """
        """
        # return new_mesh_dict, scale
        raise NotImplementedError("_build_mesh must be overridden by the subclass of Body")

    def update_position(self, time,
                        xs, ys, zs, vxs, vys, vzs,
                        ethetas, elongans, eincls,
                        ds=None, Fs=None,
                        ignore_effects=False,
                        component_com_x=None,
                        **kwargs):
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
        """
        logger.debug("{}.update_position ignore_effects={}".format(self.component, ignore_effects))
        self.reset_time(time, ethetas[self.ind_self], elongans[self.ind_self], eincls[self.ind_self])

        #-- Get current position/euler information
        # TODO: get rid of this ugly _value stuff
        pos = (_value(xs[self.ind_self]), _value(ys[self.ind_self]), _value(zs[self.ind_self]))
        vel = (_value(vxs[self.ind_self_vel]), _value(vys[self.ind_self_vel]), _value(vzs[self.ind_self_vel]))
        euler = (_value(ethetas[self.ind_self]), _value(elongans[self.ind_self]), _value(eincls[self.ind_self]))
        euler_vel = (_value(ethetas[self.ind_self_vel]), _value(elongans[self.ind_self_vel]), _value(eincls[self.ind_self_vel]))

        # TODO: eventually pass etheta to has_standard_mesh
        # TODO: implement reprojection as an option based on a nearby standard?
        if self.needs_remesh or not self.has_standard_mesh():
            logger.debug("{}.update_position: remeshing at t={}".format(self.component, time))
            # track whether we did the remesh or not, so we know if we should
            # compute local quantities if not otherwise necessary
            did_remesh = True

            # TODO: allow time dependence on d and F from dynamics
            # d = _value(ds[self.ind_self])
            # F = _value(Fs[self.ind_self])

            new_mesh_dict, scale = self._build_mesh(mesh_method=self.mesh_method)
            if self.mesh_method != 'wd':
                new_mesh_dict = self._offset_mesh(new_mesh_dict)

                # We only need the gradients where we'll compute local
                # quantities which, for a marching mesh, is at the vertices.
                new_mesh_dict['normgrads'] = new_mesh_dict.pop('vnormgrads', np.array([]))

            # And lastly, let's fill the velocities column - with zeros
            # at each of the vertices
            new_mesh_dict['velocities'] = np.zeros(new_mesh_dict['vertices'].shape if self.mesh_method != 'wd' else new_mesh_dict['centers'].shape)

            new_mesh_dict['tareas'] = np.array([])


            # TODO: need to be very careful about self.sma vs self._scale - maybe need to make a self._instantaneous_scale???
            # self._scale = scale

            if not self.has_standard_mesh():
                # then we only computed this because we didn't already have a
                # standard_mesh... so let's save this for future use
                # TODO: eventually pass etheta to save_as_standard_mesh
                protomesh = mesh.ProtoMesh(**new_mesh_dict)
                self.save_as_standard_mesh(protomesh)

            # Here we'll build a scaledprotomesh directly from the newly
            # marched mesh
            # NOTE that we're using scale from the new
            # mesh rather than self._scale since the instantaneous separation
            # has likely changed since periastron
            scaledprotomesh = mesh.ScaledProtoMesh(scale=scale, **new_mesh_dict)

        else:
            logger.debug("{}.update_position: accessing standard mesh at t={}".format(self.component, self.time))
            # track whether we did the remesh or not, so we know if we should
            # compute local quantities if not otherwise necessary
            did_remesh = False

            # We still need to go through scaledprotomesh instead of directly
            # to mesh since features may want to process the body-centric
            # coordinates before placing in orbit

            # TODO: eventually pass etheta to get_standard_mesh
            scaledprotomesh = self.get_standard_mesh(scaled=True)
            # TODO: can we avoid an extra copy here?


        if not ignore_effects and len(self.features):
            logger.debug("{}.update_position: processing features at t={}".format(self.component, self.time))
            # First allow features to edit the coords_for_computations (pvertices).
            # Changes here WILL affect future computations for logg, teff,
            # intensities, etc.  Note that these WILL NOT affect the
            # coords_for_observations automatically - those should probably be
            # perturbed as well, unless there is a good reason not to.
            for feature in self.features:
                # NOTE: these are ALWAYS done on the protomesh
                coords_for_observations = feature.process_coords_for_computations(scaledprotomesh.coords_for_computations, s=self.polar_direction_xyz, t=self.time)
                if scaledprotomesh._compute_at_vertices:
                    scaledprotomesh.update_columns(pvertices=coords_for_observations)

                else:
                    scaledprotomesh.update_columns(centers=coords_for_observations)
                    raise NotImplementedError("areas are not updated for changed mesh")


            for feature in self.features:
                coords_for_observations = feature.process_coords_for_observations(scaledprotomesh.coords_for_computations, scaledprotomesh.coords_for_observations, s=self.polar_direction_xyz, t=self.time)
                if scaledprotomesh._compute_at_vertices:
                    scaledprotomesh.update_columns(vertices=coords_for_observations)

                    # TODO [DONE?]: centers either need to be supported or we need to report
                    # vertices in the frontend as x, y, z instead of centers

                    updated_props = libphoebe.mesh_properties(scaledprotomesh.vertices,
                                                              scaledprotomesh.triangles,
                                                              tnormals=True,
                                                              areas=True)

                    scaledprotomesh.update_columns(**updated_props)

                else:
                    scaledprotomesh.update_columns(centers=coords_for_observations)
                    raise NotImplementedError("areas are not updated for changed mesh")


        # TODO NOW [OPTIMIZE]: get rid of the deepcopy here - but without it the
        # mesh velocities build-up and do terrible things.  It may be possible
        # to just clear the velocities in get_standard_mesh()?
        logger.debug("{}.update_position: placing in orbit, Mesh.from_scaledproto at t={}".format(self.component, self.time))
        self._mesh = mesh.Mesh.from_scaledproto(scaledprotomesh.copy(),
                                                pos, vel, euler, euler_vel,
                                                self.polar_direction_xyz*self.freq_rot*self._scale,
                                                component_com_x)


        # Lastly, we'll recompute physical quantities (not observables) if
        # needed for this time-step.
        # TODO [DONE?]: make sure features smartly trigger needs_recompute_instantaneous
        # TODO: get rid of the or True here... the problem is that we're saving the standard mesh before filling local quantities
        if self.needs_recompute_instantaneous or did_remesh or self._force_recompute_instantaneous_next_update_position:
            logger.debug("{}.update_position: calling compute_local_quantities at t={} ignore_effects={}".format(self.component, self.time, ignore_effects))
            self.compute_local_quantities(xs, ys, zs, ignore_effects)
            self._force_recompute_instantaneous_next_update_position = False

        return

    def compute_local_quantities(self, xs, ys, zs, ignore_effects=False, **kwargs):
        """
        """
        raise NotImplementedError("compute_local_quantities needs to be overridden by the subclass of Star")

    def populate_observable(self, time, kind, dataset, ignore_effects=False, **kwargs):
        """
        TODO: add documentation
        """

        if kind in ['mesh', 'orb']:
            return

        if time==self.time and dataset in self.populated_at_time and 'pblum' not in kind:
            # then we've already computed the needed columns

            # TODO: handle the case of intensities already computed by
            # /different/ dataset (ie RVs computed first and filling intensities
            # and then lc requesting intensities with SAME passband/atm)
            return

        new_mesh_cols = getattr(self, '_populate_{}'.format(kind.lower()))(dataset, ignore_effects=ignore_effects, **kwargs)

        for key, col in new_mesh_cols.items():

            self.mesh.update_columns_dict({'{}:{}'.format(key, dataset): col})

        self.populated_at_time.append(dataset)

class Star(Body):
    def __init__(self, component, comp_no, ind_self, ind_sibling, masses, ecc, incl,
                 long_an, t0, do_mesh_offset, mesh_init_phi,

                 atm, datasets, passband, intens_weighting,
                 extinct, Rv,
                 ld_mode, ld_func, ld_coeffs, ld_coeffs_source,
                 lp_profile_rest,
                 requiv, sma,
                 polar_direction_uvw,
                 freq_rot,
                 teff, gravb_bol, abun,
                 irrad_frac_refl,
                 mesh_method, is_single,
                 do_rv_grav,
                 features,
                 **kwargs):
        """
        """
        super(Star, self).__init__(component, comp_no, ind_self, ind_sibling,
                                   masses, ecc,
                                   incl, long_an, t0,
                                   do_mesh_offset,
                                   mesh_init_phi)

        # store everything that is needed by Star but not passed to Body
        self.requiv = requiv
        self.sma = sma
        # TODO: this may not always be the case: i.e. for single stars
        self._scale = sma

        self.polar_direction_uvw = polar_direction_uvw.astype(float)
        self.freq_rot = freq_rot
        self.teff = teff
        self.gravb_bol = gravb_bol
        self.abun = abun
        self.irrad_frac_refl = irrad_frac_refl
        self.mesh_method = mesh_method
        self.ntriangles = kwargs.get('ntriangles', 1000)                    # Marching
        self.distortion_method = kwargs.get('distortion_method', 'roche')   # Marching (WD assumes roche)
        self.gridsize = kwargs.get('gridsize', 90)                          # WD
        self.is_single = is_single
        self.atm = atm

        # DATSET-DEPENDENT DICTS
        self.passband = passband
        self.intens_weighting = intens_weighting
        self.extinct = extinct
        self.Rv = Rv
        self.ld_mode = ld_mode
        self.ld_func = ld_func
        self.ld_coeffs = ld_coeffs
        self.ld_coeffs_source = ld_coeffs_source
        self.lp_profile_rest = lp_profile_rest

        # Let's create a dictionary to handle how each dataset should scale between
        # absolute and relative intensities.
        self._pblum_scale = {}
        self._ptfarea = {}

        self.do_rv_grav = do_rv_grav
        self.features = features


    @classmethod
    def from_bundle(cls, b, component, compute=None,
                    datasets=[], **kwargs):
        """
        Build a star from the :class:`phoebe.frontend.bundle.Bundle` and its
        hierarchy.

        Usually it makes more sense to call :meth:`System.from_bundle` directly.

        :parameter b: the :class:`phoebe.frontend.bundle.Bundle`
        :parameter str component: label of the component in the bundle
        :parameter str compute: name of the computeoptions in the bundle
        :parameter list datasets: list of names of datasets
        :parameter **kwargs: temporary overrides for computeoptions
        :return: an instantiated :class:`Star` object
        """
        # TODO [DONE?]: handle overriding options from kwargs
        # TODO [DONE?]: do we need dynamics method???

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

        self_ps = b.filter(component=component, context='component', **_skip_filter_checks)
        requiv = self_ps.get_value(qualifier='requiv', unit=u.solRad, **_skip_filter_checks)


        masses = [b.get_value(qualifier='mass', component=star, context='component', unit=u.solMass, **_skip_filter_checks) for star in starrefs]
        if b.hierarchy.get_parent_of(component) is not None:
            sma = b.get_value(qualifier='sma', component=label_orbit, context='component', unit=u.solRad, **_skip_filter_checks)
            ecc = b.get_value(qualifier='ecc', component=label_orbit, context='component', **_skip_filter_checks)
            is_single = False
        else:
            # single star case
            sma = 1.0
            ecc = 0.0
            is_single = True

        incl = b.get_value(qualifier='incl', component=label_orbit, context='component', unit=u.rad, **_skip_filter_checks)
        long_an = b.get_value(qualifier='long_an', component=label_orbit, context='component', unit=u.rad, **_skip_filter_checks)

        # NOTE: these may not be used when not visible for contact systems, so
        # Star_roche_envelope_half should ignore and override with
        # aligned/synchronous
        incl_star = self_ps.get_value(qualifier='incl', unit=u.rad, **_skip_filter_checks)
        long_an_star = self_ps.get_value(qualifier='long_an', unit=u.rad, **_skip_filter_checks)
        polar_direction_uvw = mesh.spin_in_system(incl_star, long_an_star)
        # freq_rot for contacts will be provided by that subclass as 2*pi/P_orb since they're always synchronous
        freq_rot = self_ps.get_value(qualifier='freq', unit=u.rad/u.d, **_skip_filter_checks)

        t0 = b.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)

        teff = b.get_value(qualifier='teff', component=component, context='component', unit=u.K, **_skip_filter_checks)
        gravb_bol= b.get_value(qualifier='gravb_bol', component=component, context='component', **_skip_filter_checks)

        abun = b.get_value(qualifier='abun', component=component, context='component', **_skip_filter_checks)
        irrad_frac_refl = b.get_value(qualifier='irrad_frac_refl_bol', component=component, context='component', **_skip_filter_checks)

        try:
            rv_grav_override = kwargs.pop('rv_grav', None)
            do_rv_grav = b.get_value(qualifier='rv_grav', component=component, compute=compute, rv_grav=rv_grav_override, **_skip_filter_checks) if compute is not None else False
        except ValueError:
            # rv_grav may not have been copied to this component if no rvs are attached
            do_rv_grav = False

        if b.hierarchy.is_meshable(component):
            mesh_method_override = kwargs.pop('mesh_method', None)
            mesh_method = b.get_value(qualifier='mesh_method', component=component, compute=compute, mesh_method=mesh_method_override, **_skip_filter_checks) if compute is not None else 'marching'

            if mesh_method == 'marching':
                # we need check_visible=False in each of these in case mesh_method
                # was overriden from kwargs
                ntriangles_override = kwargs.pop('ntriangles', None)
                kwargs['ntriangles'] = b.get_value(qualifier='ntriangles', component=component, compute=compute, ntriangles=ntriangles_override, **_skip_filter_checks) if compute is not None else 1000
                distortion_method_override = kwargs.pop('distortion_method', None)
                kwargs['distortion_method'] = b.get_value(qualifier='distortion_method', component=component, compute=compute, distortion_method=distortion_method_override, **_skip_filter_checks) if compute is not None else distortion_method_override if distortion_method_override is not None else 'roche'
            elif mesh_method == 'wd':
                # we need check_visible=False in each of these in case mesh_method
                # was overriden from kwargs
                gridsize_override = kwargs.pop('gridsize', None)
                kwargs['gridsize'] = b.get_value(qualifier='gridsize', component=component, compute=compute, gridsize=gridsize_override, **_skip_filter_checks) if compute is not None else 30
            else:
                raise NotImplementedError
        else:
            # then we're half of a contact... the Envelope object will handle meshing
            mesh_method = kwargs.pop('mesh_method', None)

        features = []
        for feature in b.filter(qualifier='enabled', compute=compute, value=True, **_skip_filter_checks).features:
            feature_ps = b.get_feature(feature=feature, **_skip_filter_checks)
            if feature_ps.component != component:
                continue
            feature_cls = globals()[feature_ps.kind.title()]
            features.append(feature_cls.from_bundle(b, feature))

        if conf.devel:
            mesh_offset_override = kwargs.pop('mesh_offset', None)
            try:
                do_mesh_offset = b.get_value(qualifier='mesh_offset', compute=compute, mesh_offset=mesh_offset_override, **_skip_filter_checks)
            except ValueError:
                do_mesh_offset = mesh_offset_override
        else:
            do_mesh_offset = True

        if conf.devel and mesh_method=='marching' and compute is not None:
            kwargs.setdefault('mesh_init_phi', b.get_value(qualifier='mesh_init_phi', compute=compute, component=component, unit=u.rad, mesh_init_phi=kwargs.get('mesh_init_phi', None), **_skip_filter_checks))

        datasets_intens = [ds for ds in b.filter(kind=['lc', 'rv', 'lp'], context='dataset').datasets if ds != '_default']
        datasets_lp = [ds for ds in b.filter(kind='lp', context='dataset', **_skip_filter_checks).datasets if ds != '_default']
        atm_override = kwargs.pop('atm', None)
        if isinstance(atm_override, dict):
            atm_override = atm_override.get(component, None)
        atm = b.get_value(qualifier='atm', compute=compute, component=component, atm=atm_override, **_skip_filter_checks) if compute is not None else atm_override if atm_override is not None else 'ck2004'
        passband_override = kwargs.pop('passband', None)
        passband = {ds: b.get_value(qualifier='passband', dataset=ds, passband=passband_override, **_skip_filter_checks) for ds in datasets_intens}
        intens_weighting_override = kwargs.pop('intens_weighting', None)
        intens_weighting = {ds: b.get_value(qualifier='intens_weighting', dataset=ds, intens_weighting=intens_weighting_override, **_skip_filter_checks) for ds in datasets_intens}
        ebv_override = kwargs.pop('ebv', None)
        extinct = b.get_value('ebv', context='system', ebv=ebv_override, **_skip_filter_checks)
        Rv_override = kwargs.pop('Rv', None)
        Rv = b.get_value('Rv', context='system', Rv=Rv_override)
        ld_mode_override = kwargs.pop('ld_mode', None)
        ld_mode = {ds: b.get_value(qualifier='ld_mode', dataset=ds, component=component, ld_mode=ld_mode_override, **_skip_filter_checks) for ds in datasets_intens}
        ld_func_override = kwargs.pop('ld_func', None)
        ld_func = {ds: b.get_value(qualifier='ld_func', dataset=ds, component=component, ld_func=ld_func_override, **_skip_filter_checks) for ds in datasets_intens}
        ld_coeffs_override = kwargs.pop('ld_coeffs', None)
        ld_coeffs = {ds: b.get_value(qualifier='ld_coeffs', dataset=ds, component=component, context='dataset', ld_coeffs=ld_coeffs_override, **_skip_filter_checks) for ds in datasets_intens}
        ld_coeffs_source_override = kwargs.pop('ld_coeffs_source', None)
        ld_coeffs_source = {ds: b.get_value(qualifier='ld_coeffs_source', dataset=ds, component=component, ld_coeffs_source=ld_coeffs_source_override, **_skip_filter_checks) for ds in datasets_intens}
        ld_func_bol_override = kwargs.pop('ld_func_bol', None)
        ld_func['bol'] = b.get_value(qualifier='ld_func_bol', component=component, context='component', ld_func_bol=ld_func_bol_override, **_skip_filter_checks)
        ld_coeffs_bol_override = kwargs.pop('ld_coeffs_bol', None)
        ld_coeffs['bol'] = b.get_value(qualifier='ld_coeffs_bol', component=component, context='component', ld_coeffs_bol=ld_coeffs_bol_override, **_skip_filter_checks)
        profile_rest_override = kwargs.pop('profile_rest', None)
        lp_profile_rest = {ds: b.get_value(qualifier='profile_rest', dataset=ds, unit=u.nm, profile_rest=profile_rest_override, **_skip_filter_checks) for ds in datasets_lp}


        # we'll pass kwargs on here so they can be overridden by the classmethod
        # of any subclass and then intercepted again by the __init__ by the
        # same subclass.  Note: kwargs also hold meshing kwargs which are used
        # by Star.__init__
        return cls(component, comp_no, ind_self, ind_sibling,
                   masses, ecc,
                   incl, long_an, t0,
                   do_mesh_offset,
                   kwargs.pop('mesh_init_phi', 0.0),

                   atm,
                   datasets,
                   passband,
                   intens_weighting,
                   extinct, Rv,
                   ld_mode,
                   ld_func,
                   ld_coeffs,
                   ld_coeffs_source,
                   lp_profile_rest,
                   requiv,
                   sma,
                   polar_direction_uvw,
                   freq_rot,
                   teff,
                   gravb_bol,
                   abun,
                   irrad_frac_refl,
                   mesh_method,
                   is_single,
                   do_rv_grav,
                   features,
                   **kwargs
                   )

    @property
    def is_convex(self):
        """
        """
        # in general this is False, subclasses can override this to True
        # if they can guarantee that their mesh will be strictly convex
        return False

    @property
    def needs_recompute_instantaneous(self):
        """
        whether the Body needs local quantities recomputed at each time, even
        if needs_remesh == False (instantaneous local quantities will be recomputed
        if needs_remesh=True, whether or not this is True)

        this should be overridden by any subclass of Star, if necessary
        """
        return True

    @property
    def needs_remesh(self):
        """
        whether the star needs to be re-meshed (for any reason)
        """
        return True

    @property
    def is_misaligned(self):
        """
        whether the star is misaligned wrt its orbit.  This probably does not
        need to be overridden by subclasses, but can be useful to use within
        the overriden methods for needs_remesh and needs_recompute_instantaneous
        """
        # should be defined for any class that subclasses Star that supports
        # misalignment
        if self.is_single:
            return False

        return self.polar_direction_xyz[2] != 1.0

    @property
    def spots(self):
        return [f for f in self.features if f.__class__.__name__=='Spot']

    @property
    def polar_direction_xyz(self):
        """
        get current polar direction in Roche (xyz) coordinates
        """
        return mesh.spin_in_roche(self.polar_direction_uvw,
                                  self.true_anom, self.elongan, self.eincl).astype(float)

    def get_target_volume(self, etheta=0.0, scaled=False):
        """
        TODO: add documentation

        get the volume that the Star should have at a given euler theta
        """
        # TODO: make this a function of d instead of etheta?
        logger.debug("determining target volume at t={}, theta={}".format(self.time, etheta))

        # TODO: eventually this could allow us to "break" volume conservation
        # and have volume be a function of d, with some scaling factor provided
        # by the user as a parameter.  Until then, we'll assume volume is
        # conserved which means the volume should always be the same
        volume = 4./3 * np.pi * self.requiv**3

        if not scaled:
            return volume / self._scale**3
        else:
            return volume

    @property
    def north_pole_uvw(self):
        """location of the north pole in the global/system frame"""
        # TODO: is this rpole scaling true for all distortion_methods??
        rpole = self.instantaneous_rpole*self.sma
        return self.polar_direction_uvw*rpole+self.mesh._pos

    def _build_mesh(self, *args, **kwargs):
        """
        """
        # return new_mesh_dict, scale
        raise NotImplementedError("_build_mesh must be overridden by the subclass of Star")

    def compute_local_quantities(self, xs, ys, zs, ignore_effects=False, **kwargs):
        # Now fill local instantaneous quantities
        self._fill_loggs(ignore_effects=ignore_effects)
        self._fill_gravs()
        self._fill_teffs(ignore_effects=ignore_effects)
        self._fill_abuns(abun=self.abun)
        self._fill_albedos(irrad_frac_refl=self.irrad_frac_refl)

    @property
    def _rpole_func(self):
        """
        """
        # the provided function must take *self.instantaneous_mesh_args as the
        # only arguments.  If this is not the case, the subclass must also override
        # instantaneous_rpole
        # pole_func = getattr(libphoebe, '{}_pole'.format('{}_misaligned'.format(self.distortion_method) if self.distortion_method in ['roche', 'rotstar'] else self.distortion_method))
        raise NotImplementedError("rpole_func must be overriden by the subclass of Star")

    @property
    def _gradOmega_func(self):
        """
        """
        # the provided function must take *self.instantaneous_mesh_args as the
        # only arguments.  If this is not the case, the subclass must also override
        # instantaneous_gpole
        # gradOmega_func = getattr(libphoebe, '{}_gradOmega_only'.format('{}_misaligned'.format(self.distortion_method) if self.distortion_method in ['roche', 'rotstar'] else self.distortion_method))
        raise NotImplementedError("gradOmega_func must be overriden by the subclass of Star")

    @property
    def instantaneous_d(self):
        logger.debug("{}.instantaneous_d".format(self.component))

        if 'd' not in self.inst_vals.keys():
            logger.debug("{}.instantaneous_d COMPUTING".format(self.component))

            self.inst_vals['d'] = np.sqrt(sum([(_value(self._get_coords_by_index(c, self.ind_self)) -
                                             _value(self._get_coords_by_index(c, self.ind_sibling)))**2
                                             for c in (self.system.xs,self.system.ys,self.system.zs)])) / self._scale

        return self.inst_vals['d']

    @property
    def instantaneous_rpole(self):
        # NOTE: unscaled... should we make this a get_instantaneous_rpole(scaled=False)?
        logger.debug("{}.instantaneous_rpole".format(self.component))

        if 'rpole' not in self.inst_vals.keys():
            logger.debug("{}.instantaneous_rpole COMPUTING".format(self.component))

            self.inst_vals['rpole'] = self._rpole_func(*self.instantaneous_mesh_args)

        return self.inst_vals['rpole']

    @property
    def instantaneous_gpole(self):
        logger.debug("{}.instantaneous_gpole".format(self.component))

        if 'gpole' not in self.inst_vals.keys():
            logger.debug("{}.instantaneous_gpole COMPUTING".format(self.component))

            rpole_ = np.array([0., 0., self.instantaneous_rpole])

            # TODO: this is a little ugly as it assumes Phi is the last argument in mesh_args
            args = list(self.instantaneous_mesh_args)[:-1]+[rpole_]
            grads = self._gradOmega_func(*args)  # needs choice=0/1 for contacts?
            gpole = np.linalg.norm(grads)

            self.inst_vals['gpole'] = gpole * g_rel_to_abs(self.masses[self.ind_self], self.sma)

        return self.inst_vals['gpole']

    @property
    def instantaneous_tpole(self):
        """
        compute the instantaenous temperature at the pole to achieve the mean
        effective temperature (teff) provided by the user
        """
        logger.debug("{}.instantaneous_tpole".format(self.component))

        if 'tpole' not in self.inst_vals.keys():
            logger.debug("{}.instantaneous_tpole COMPUTING".format(self.component))

            if self.mesh is None:
                raise ValueError("mesh must be computed before determining tpole")
            # Convert from mean to polar by dividing flux by gravity darkened flux (Ls drop out)
            # see PHOEBE Legacy scientific reference eq 5.20
            self.inst_vals['tpole'] = self.teff*(np.sum(self.mesh.areas) / np.sum(self.mesh.gravs.centers*self.mesh.areas))**(0.25)

        return self.inst_vals['tpole']

    @property
    def instantaneous_mesh_args(self):
        """
        determine instantaneous parameters needed for meshing
        """
        raise NotImplementedError("instantaneous_mesh_args must be overridden by the subclass of Sar")

    def _fill_loggs(self, mesh=None, ignore_effects=False):
        """
        TODO: add documentation

        Calculate local surface gravity

        GMSunNom = 1.3271244e20 m**3 s**-2
        RSunNom = 6.597e8 m
        """
        logger.debug("{}._fill_loggs".format(self.component))

        if mesh is None:
            mesh = self.mesh

        loggs = np.log10(mesh.normgrads.for_computations * g_rel_to_abs(self.masses[self.ind_self], self.sma))

        if not ignore_effects:
            for feature in self.features:
                if feature.proto_coords:
                    loggs = feature.process_loggs(loggs, mesh.roche_coords_for_computations, s=self.polar_direction_xyz, t=self.time)
                else:
                    loggs = feature.process_loggs(loggs, mesh.coords_for_computations, s=self.polar_direction_xyz, t=self.time)

        mesh.update_columns(loggs=loggs)

        if not self.needs_recompute_instantaneous:
            logger.debug("{}._fill_loggs: copying loggs to standard mesh".format(self.component))
            theta = 0.0
            self._standard_meshes[theta].update_columns(loggs=loggs)

    def _fill_gravs(self, mesh=None, **kwargs):
        """
        TODO: add documentation

        requires _fill_loggs to have been called
        """
        logger.debug("{}._fill_gravs".format(self.component))

        if mesh is None:
            mesh = self.mesh

        # TODO: rename 'gravs' to 'gdcs' (gravity darkening corrections)

        gravs = ((mesh.normgrads.for_computations * g_rel_to_abs(self.masses[self.ind_self], self.sma))/self.instantaneous_gpole)**self.gravb_bol

        mesh.update_columns(gravs=gravs)

        if not self.needs_recompute_instantaneous:
            logger.debug("{}._fill_gravs: copying gravs to standard mesh".format(self.component))
            theta = 0.0
            self._standard_meshes[theta].update_columns(gravs=gravs)


    def _fill_teffs(self, mesh=None, ignore_effects=False, **kwargs):
        r"""

        requires _fill_loggs and _fill_gravs to have been called

        Calculate local temperature of a Star.
        """
        logger.debug("{}._fill_teffs".format(self.component))

        if mesh is None:
            mesh = self.mesh

        # Now we can compute the local temperatures.
        # see PHOEBE Legacy scientific reference eq 5.23
        teffs = self.instantaneous_tpole*mesh.gravs.for_computations**0.25

        if not ignore_effects:
            for feature in self.features:
                if feature.proto_coords:

                    if self.__class__.__name__ == 'Star_roche_envelope_half' and self.ind_self != self.ind_self_vel:
                        # then this is the secondary half of a contact envelope
                        roche_coords_for_computations = np.array([1.0, 0.0, 0.0]) - mesh.roche_coords_for_computations
                    else:
                        roche_coords_for_computations = mesh.roche_coords_for_computations
                    teffs = feature.process_teffs(teffs, roche_coords_for_computations, s=self.polar_direction_xyz, t=self.time)
                else:
                    teffs = feature.process_teffs(teffs, mesh.coords_for_computations, s=self.polar_direction_xyz, t=self.time)

        mesh.update_columns(teffs=teffs)

        if not self.needs_recompute_instantaneous:
            logger.debug("{}._fill_teffs: copying teffs to standard mesh".format(self.component))
            theta = 0.0
            self._standard_meshes[theta].update_columns(teffs=teffs)

    def _fill_abuns(self, mesh=None, abun=0.0):
        """
        TODO: add documentation
        """
        logger.debug("{}._fill_abuns".format(self.component))

        if mesh is None:
            mesh = self.mesh

        # TODO: support from frontend

        mesh.update_columns(abuns=abun)

        if not self.needs_recompute_instantaneous:
            logger.debug("{}._fill_abuns: copying abuns to standard mesh".format(self.component))
            theta = 0.0
            self._standard_meshes[theta].update_columns(abuns=abun)

    def _fill_albedos(self, mesh=None, irrad_frac_refl=0.0):
        """
        TODO: add documentation
        """
        logger.debug("{}._fill_albedos".format(self.component))

        if mesh is None:
            mesh = self.mesh

        mesh.update_columns(irrad_frac_refl=irrad_frac_refl)

        if not self.needs_recompute_instantaneous:
            logger.debug("{}._fill_albedos: copying albedos to standard mesh".format(self.component))
            theta = 0.0
            self._standard_meshes[theta].update_columns(irrad_frac_refl=irrad_frac_refl)

    def compute_luminosity(self, dataset, scaled=True, **kwargs):
        """
        """
        # assumes dataset-columns have already been populated
        logger.debug("{}.compute_luminosity(dataset={})".format(self.component, dataset))

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
        if scaled:
            return abs_luminosity * self.get_pblum_scale(dataset)
        else:
            return abs_luminosity

    def set_pblum_scale(self, dataset, pblum_scale, **kwargs):
        """
        """
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

    def _populate_lp(self, dataset, **kwargs):
        """
        Populate columns necessary for an LP dataset

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`
        """
        logger.debug("{}._populate_lp(dataset={})".format(self.component, dataset))

        profile_rest = kwargs.get('profile_rest', self.lp_profile_rest.get(dataset))

        rv_cols = self._populate_rv(dataset, **kwargs)

        cols = rv_cols
        # rvs = (rv_cols['rvs']*u.solRad/u.d).to(u.m/u.s).value
        # cols['dls'] = rv_cols['rvs']*profile_rest/c.c.si.value

        return cols

    def _populate_rv(self, dataset, **kwargs):
        """
        Populate columns necessary for an RV dataset

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`
        """
        logger.debug("{}._populate_rv(dataset={})".format(self.component, dataset))

        # We need to fill all the flux-related columns so that we can weigh each
        # triangle's rv by its flux in the requested passband.
        lc_cols = self._populate_lc(dataset, **kwargs)

        # rv per element is just the z-component of the velocity vectory.  Note
        # the change in sign from our right-handed system to rv conventions.
        # These will be weighted by the fluxes when integrating

        rvs = -1*self.mesh.velocities.for_computations[:,2]

        # Gravitational redshift
        if self.do_rv_grav:
            # self.mass is in solar masses
            # self.instantaneous_rpole is in Roche units, i.e. r/a
            rpole = self.instantaneous_rpole*self.sma*u.solRad
            rv_grav = c.G*(self.mass*u.solMass)/rpole/c.c

            # rvs are in solRad/d internally, so we need to convert:
            rv_grav = rv_grav.to('solRad/d').value

            rvs += rv_grav

        cols = lc_cols
        cols['rvs'] = rvs
        return cols


    def _populate_lc(self, dataset, ignore_effects=False, **kwargs):
        """
        Populate columns necessary for an LC dataset

        This should not be called directly, but rather via :meth:`Body.populate_observable`
        or :meth:`System.populate_observables`

        :raises NotImplementedError: if lc_method is not supported
        """
        logger.debug("{}._populate_lc(dataset={}, ignore_effects={})".format(self.component, dataset, ignore_effects))

        lc_method = kwargs.get('lc_method', 'numerical')  # TODO: make sure this is actually passed

        passband = kwargs.get('passband', self.passband.get(dataset, None))
        intens_weighting = kwargs.get('intens_weighting', self.intens_weighting.get(dataset, None))
        atm = kwargs.get('atm', self.atm)
        extinct = kwargs.get('extinct', self.extinct)
        Rv = kwargs.get('Rv', self.Rv)
        ld_mode = kwargs.get('ld_mode', self.ld_mode.get(dataset, None))
        ld_func = kwargs.get('ld_func', self.ld_func.get(dataset, None))
        ld_coeffs = kwargs.get('ld_coeffs', self.ld_coeffs.get(dataset, None)) if ld_mode == 'manual' else None
        ld_coeffs_source = kwargs.get('ld_coeffs_source', self.ld_coeffs_source.get(dataset, 'none')) if ld_mode == 'lookup' else None
        if ld_mode == 'interp':
            # calls to pb.Imu need to pass on ld_func='interp'
            # NOTE: we'll do another check when calling pb.Imu, but we'll also
            # change the value here for the debug logger
            ld_func = 'interp'
            ldatm = atm
        elif ld_mode == 'lookup':
            if ld_coeffs_source == 'auto':
                if atm == 'blackbody':
                    ldatm = 'ck2004'
                elif atm == 'extern_atmx':
                    ldatm = 'ck2004'
                elif atm == 'extern_planckint':
                    ldatm = 'ck2004'
                else:
                    ldatm = atm
            else:
                ldatm = ld_coeffs_source
        elif ld_mode == 'manual':
            ldatm = 'none'
        else:
            raise NotImplementedError

        boosting_method = kwargs.get('boosting_method', self.boosting_method)

        logger.debug("ld_func={}, ld_coeffs={}, atm={}, ldatm={}".format(ld_func, ld_coeffs, atm, ldatm))

        pblum = kwargs.get('pblum', 4*np.pi)

        if lc_method=='numerical':

            pb = passbands.get_passband(passband)

            if ldatm != 'none' and '{}:ldint'.format(ldatm) not in pb.content:
                if ld_mode == 'lookup':
                    raise ValueError("{} not supported for limb-darkening with {}:{} passband.  Try changing the value of the ld_coeffs_source parameter".format(ldatm, pb.pbset, pb.pbname))
                else:
                    raise ValueError("{} not supported for limb-darkening with {}:{} passband.  Try changing the value of the atm parameter".format(ldatm, pb.pbset, pb.pbname))

            if intens_weighting=='photon':
                ptfarea = pb.ptf_photon_area/pb.h/pb.c
            else:
                ptfarea = pb.ptf_area

            self.set_ptfarea(dataset, ptfarea)

            try:
                ldint = pb.ldint(Teff=self.mesh.teffs.for_computations,
                                 logg=self.mesh.loggs.for_computations,
                                 abun=self.mesh.abuns.for_computations,
                                 ldatm=ldatm,
                                 ld_func=ld_func if ld_mode != 'interp' else ld_mode,
                                 ld_coeffs=ld_coeffs,
                                 photon_weighted=intens_weighting=='photon')
            except ValueError as err:
                if str(err).split(":")[0] == 'Atmosphere parameters out of bounds':
                    # let's override with a more helpful error message
                    logger.warning(str(err))
                    if atm=='blackbody':
                        raise ValueError("Could not compute ldint with ldatm='{}'.  Try changing ld_coeffs_source to a table that covers a sufficient range of values or set ld_mode to 'manual' and manually provide coefficients via ld_coeffs. Enable 'warning' logger to see out-of-bound arrays.".format(ldatm))
                    else:
                        if ld_mode=='interp':
                            raise ValueError("Could not compute ldint with ldatm='{}'.  Try changing atm to a table that covers a sufficient range of values.  If necessary, set atm to 'blackbody' and/or ld_mode to 'manual' (in which case coefficients will need to be explicitly provided via ld_coeffs). Enable 'warning' logger to see out-of-bound arrays.".format(ldatm))
                        elif ld_mode == 'lookup':
                            raise ValueError("Could not compute ldint with ldatm='{}'.  Try changing atm to a table that covers a sufficient range of values.  If necessary, set atm to 'blackbody' and/or ld_mode to 'manual' (in which case coefficients will need to be explicitly provided via ld_coeffs). Enable 'warning' logger to see out-of-bound arrays.".format(ldatm))
                        else:
                            # manual... this means that the atm itself is out of bounds, so the only option is atm=blackbody
                            raise ValueError("Could not compute ldint with ldatm='{}'.  Try changing atm to a table that covers a sufficient range of values.  If necessary, set atm to 'blackbody', ld_mode to 'manual', and provide coefficients via ld_coeffs. Enable 'warning' logger to see out-of-bound arrays.".format(ldatm))
                else:
                    raise err

            try:
                # abs_normal_intensities are the normal emergent passband intensities:
                abs_normal_intensities = pb.Inorm(Teff=self.mesh.teffs.for_computations,
                                                  logg=self.mesh.loggs.for_computations,
                                                  abun=self.mesh.abuns.for_computations,
                                                  atm=atm,
                                                  ldatm=ldatm,
                                                  ldint=ldint,
                                                  photon_weighted=intens_weighting=='photon')
            except ValueError as err:
                if str(err).split(":")[0] == 'Atmosphere parameters out of bounds':
                    # let's override with a more helpful error message
                    logger.warning(str(err))
                    raise ValueError("Could not compute intensities with atm='{}'.  Try changing atm to a table that covers a sufficient range of values (or to 'blackbody' in which case ld_mode will need to be set to 'manual' and coefficients provided via ld_coeffs).  Enable 'warning' logger to see out-of-bounds arrays.".format(atm))
                else:
                    raise err

            # abs_intensities are the projected (limb-darkened) passband intensities
            # TODO: why do we need to use abs(mus) here?
            # ! Because the interpolation within Imu will otherwise fail.
            # ! It would be best to pass only [visibilities > 0] elements to Imu.
            abs_intensities = pb.Imu(Teff=self.mesh.teffs.for_computations,
                                     logg=self.mesh.loggs.for_computations,
                                     abun=self.mesh.abuns.for_computations,
                                     mu=abs(self.mesh.mus_for_computations),
                                     atm=atm,
                                     ldatm=ldatm,
                                     ldint=ldint,
                                     ld_func=ld_func if ld_mode != 'interp' else ld_mode,
                                     ld_coeffs=ld_coeffs,
                                     photon_weighted=intens_weighting=='photon')


            # Beaming/boosting
            if boosting_method == 'none' or ignore_effects:
                boost_factors = 1.0
            elif boosting_method == 'linear':
                logger.debug("calling pb.bindex for boosting_method='linear'")
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

            if extinct == 0.0:
                extinct_factors = 1.0
            else:
                extinct_factors = pb.interpolate_extinct(Teff=self.mesh.teffs.for_computations,
                                                         logg=self.mesh.loggs.for_computations,
                                                         abun=self.mesh.abuns.for_computations,
                                                         extinct=extinct,
                                                         Rv=Rv,
                                                         atm=atm,
                                                         photon_weighted=intens_weighting=='photon')

                # extinction is NOT aspect dependent, so we'll correct both
                # normal and directional intensities
                abs_intensities *= extinct_factors
                abs_normal_intensities *= extinct_factors

            # Handle pblum - distance and l3 scaling happens when integrating (in observe)
            # we need to scale each triangle so that the summed normal_intensities over the
            # entire star is equivalent to pblum / 4pi
            normal_intensities = abs_normal_intensities * self.get_pblum_scale(dataset)
            intensities = abs_intensities * self.get_pblum_scale(dataset)

        elif lc_method=='analytical':
            raise NotImplementedError("analytical fluxes not yet supported")
            # TODO: this probably needs to be moved into observe or backends.phoebe
            # (assuming it doesn't result in per-triangle quantities)

        else:
            raise NotImplementedError("lc_method '{}' not recognized".format(lc_method))

        # TODO: do we really need to store all of these if store_mesh==False?
        # Can we optimize by only returning the essentials if we know we don't need them?
        return {'abs_normal_intensities': abs_normal_intensities,
                'normal_intensities': normal_intensities,
                'abs_intensities': abs_intensities,
                'intensities': intensities,
                'ldint': ldint,
                'boost_factors': boost_factors}


class Star_roche(Star):
    """
    detached case only
    """
    def __init__(self, component, comp_no, ind_self, ind_sibling,
                 masses, ecc, incl,
                 long_an, t0, do_mesh_offset, mesh_init_phi,

                 atm, datasets, passband, intens_weighting,
                 extinct, Rv,
                 ld_mode, ld_func, ld_coeffs, ld_coeffs_source,
                 lp_profile_rest,
                 requiv, sma,
                 polar_direction_uvw,
                 freq_rot,
                 teff, gravb_bol, abun,
                 irrad_frac_refl,
                 mesh_method, is_single,
                 do_rv_grav,
                 features,

                 **kwargs):
        """
        """
        # extra things (not used by Star) will be stored in kwargs
        self.F = kwargs.pop('F', 1.0)

        super(Star_roche, self).__init__(component, comp_no, ind_self, ind_sibling,
                                         masses, ecc, incl,
                                         long_an, t0,
                                         do_mesh_offset, mesh_init_phi,

                                         atm, datasets, passband, intens_weighting,
                                         extinct, Rv,
                                         ld_mode, ld_func, ld_coeffs, ld_coeffs_source,
                                         lp_profile_rest,
                                         requiv, sma,
                                         polar_direction_uvw,
                                         freq_rot,
                                         teff, gravb_bol, abun,
                                         irrad_frac_refl,
                                         mesh_method, is_single,
                                         do_rv_grav,
                                         features,
                                         **kwargs)

    @classmethod
    def from_bundle(cls, b, component, compute=None,
                    datasets=[], **kwargs):

        self_ps = b.filter(component=component, context='component', **_skip_filter_checks)
        F = self_ps.get_value(qualifier='syncpar', **_skip_filter_checks)

        return super(Star_roche, cls).from_bundle(b, component, compute,
                                                  datasets,
                                                  F=F, **kwargs)


    @property
    def is_convex(self):
        return True

    @property
    def needs_recompute_instantaneous(self):
        # recompute instantaneous for asynchronous spots, even if meshing
        # doesn't need to be recomputed
        return self.needs_remesh or (len(self.features) and self.F != 1.0)

    @property
    def needs_remesh(self):
        """
        whether the star needs to be re-meshed (for any reason)
        """
        # TODO: may be able to get away with removing the features check and just doing for pulsations, etc?
        # TODO: what about dpdt, deccdt, dincldt, etc?

        for feature in self.features:
            if feature._remeshing_required:
                return True

        return self.is_misaligned or self.ecc != 0 or self.dynamics_method != 'keplerian'

    @property
    def _rpole_func(self):
        """
        """
        # the provided function must take *self.instantaneous_mesh_args as the
        # only arguments.  If this is not the case, the subclass must also override
        # instantaneous_rpole
        return getattr(libphoebe, 'roche_misaligned_pole')

    @property
    def _gradOmega_func(self):
        """
        """
        # the provided function must take *self.instantaneous_mesh_args as the
        # only arguments.  If this is not the case, the subclass must also override
        # instantaneous_gpole
        return getattr(libphoebe, 'roche_misaligned_gradOmega_only')

    @property
    def instantaneous_mesh_args(self):
        logger.debug("{}.instantaneous_mesh_args".format(self.component))

        if 'mesh_args' not in self.inst_vals.keys():
            logger.debug("{}.instantaneous_mesh_args COMPUTING".format(self.component))

            # self.q is automatically flipped to be 1./q for secondary components
            q = np.float64(self.q)

            F = np.float64(self.F)

            d = np.float64(self.instantaneous_d)

            # polar_direction_xyz is instantaneous based on current true_anom
            s = self.polar_direction_xyz

            # NOTE: if we ever want to break volume conservation in time,
            # get_target_volume will need to take time or true anomaly
            target_volume = np.float64(self.get_target_volume(scaled=False))
            instantaneous_vcrit = libphoebe.roche_misaligned_critical_volume(q, F, d, s)

            logger.debug("libphoebe.roche_misaligned_critical_volume(q={}, F={}, d={}, s={}) => {}".format(q, F, d, s, instantaneous_vcrit))
            if target_volume > instantaneous_vcrit:
                if abs(target_volume - instantaneous_vcrit) / target_volume < 1e-10:
                    logger.warning("target_volume of {} slightly over critical value, likely due to numerics: setting to critical value of {}".format(target_volume, instantaneous_vcrit))
                    target_volume = instantaneous_vcrit
                else:
                    raise ValueError("volume is exceeding critical value")

            logger.debug("libphoebe.roche_misaligned_Omega_at_vol(vol={}, q={}, F={}, d={}, s={})".format(target_volume, q, F, d, s))

            Phi = libphoebe.roche_misaligned_Omega_at_vol(target_volume,
                                                          q, F, d, s.astype(np.float64))

            logger.debug("libphoebe.roche_misaligned_Omega_at_vol(vol={}, q={}, F={}, d={}, s={}) => {}".format(target_volume, q, F, d, s, Phi))

            # this is assuming that we're in the reference frame of our current star,
            # so we don't need to worry about flipping Phi for the secondary.

            self.inst_vals['mesh_args'] = q, F, d, s, Phi

        return self.inst_vals['mesh_args']

    def _build_mesh(self, mesh_method, **kwargs):
        """
        this function takes mesh_method and kwargs that came from the generic Body.intialize_mesh and returns
        the grid... intialize mesh then takes care of filling columns and rescaling to the correct units, etc
        """

        # need the sma to scale between Roche and real units
        sma = kwargs.get('sma', self.sma)  # Rsol (same units as coordinates)

        mesh_args = self.instantaneous_mesh_args

        if mesh_method == 'marching':
            # TODO: do this during mesh initialization only and then keep delta fixed in time??
            ntriangles = kwargs.get('ntriangles', self.ntriangles)

            # we need the surface area of the lobe to estimate the correct value
            # to pass for delta to marching.  We will later need the volume to
            # expose its value
            logger.debug("libphoebe.roche_misaligned_area_volume{}".format(mesh_args))
            av = libphoebe.roche_misaligned_area_volume(*mesh_args,
                                                        choice=0,
                                                        larea=True,
                                                        lvolume=True)

            delta = _estimate_delta(ntriangles, av['larea'])

            logger.debug("libphoebe.roche_misaligned_marching_mesh{}".format(mesh_args))
            try:
                new_mesh = libphoebe.roche_misaligned_marching_mesh(*mesh_args,
                                                                    delta=delta,
                                                                    choice=0,
                                                                    full=True,
                                                                    max_triangles=int(ntriangles*1.5),
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
                                                                    init_phi=kwargs.get('mesh_init_phi', self.mesh_init_phi))
            except Exception as err:
                if str(err) == 'There are too many triangles!':
                    mesh_init_phi_attempts = kwargs.get('mesh_init_phi_attempts', 1) + 1
                    if mesh_init_phi_attempts > 5:
                        raise err

                    mesh_init_phi = np.random.random()*2*np.pi
                    logger.warning("mesh failed to converge, trying attempt #{} with mesh_init_phi={}".format(mesh_init_phi_attempts, mesh_init_phi))
                    kwargs['mesh_init_phi_attempts'] = mesh_init_phi_attempts
                    kwargs['mesh_init_phi'] = mesh_init_phi
                    return self._build_mesh(mesh_method, **kwargs)
                else:
                    raise err


            # In addition to the values exposed by the mesh itself, let's report
            # the volume and surface area of the lobe.  The lobe area is used
            # if mesh_offseting is required, and the volume is optionally exposed
            # to the user.
            new_mesh['volume'] = av['lvolume']  # * sma**3
            new_mesh['area'] = av['larea']      # * sma**2

            scale = sma

        elif mesh_method == 'wd':
            if self.is_misaligned:
                raise NotImplementedError("misaligned orbits not suported by mesh_method='wd'")

            N = int(kwargs.get('gridsize', self.gridsize))

            # unpack mesh_args so we can ignore s
            q, F, d, s, Phi = mesh_args

            logger.debug("mesh_wd.discretize_wd_style(N={}, q={}, F={}, d={}, Phi={})".format(N, q, F, d, Phi))
            the_grid = mesh_wd.discretize_wd_style(N, q, F, d, Phi)
            new_mesh = mesh.wd_grid_to_mesh_dict(the_grid, q, F, d)
            scale = sma

        else:
            raise NotImplementedError("mesh_method '{}' is not supported".format(mesh_method))

        return new_mesh, scale

class Star_roche_envelope_half(Star):
    def __init__(self, component, comp_no, ind_self, ind_sibling,
                 masses, ecc, incl,
                 long_an, t0, do_mesh_offset, mesh_init_phi,

                 atm, datasets, passband, intens_weighting,
                 extinct, Rv,
                 ld_mode, ld_func, ld_coeffs, ld_coeffs_source,
                 lp_profile_rest,
                 requiv, sma,
                 polar_direction_uvw,
                 freq_rot,
                 teff, gravb_bol, abun,
                 irrad_frac_refl,
                 mesh_method, is_single,
                 do_rv_grav,
                 features,

                 **kwargs):
        """
        """
        self.F = 1 # frontend run_checks makes sure that contacts are synchronous
        self.pot = kwargs.get('pot')
        # requiv won't be used, instead we'll use potential, but we'll allow
        # accessing and passing requiv anyways.

        # for contacts the secondary is on the reverse side of the roche coordinates
        # and so actually needs to be put in orbit as if it were the primary.
        super(Star_roche_envelope_half, self).__init__(component, comp_no, 0, 1,
                                         masses, ecc, incl,
                                         long_an, t0,
                                         do_mesh_offset, mesh_init_phi,

                                         atm, datasets, passband, intens_weighting,
                                         extinct, Rv,
                                         ld_mode, ld_func, ld_coeffs, ld_coeffs_source,
                                         lp_profile_rest,
                                         requiv, sma,
                                         polar_direction_uvw,
                                         freq_rot,
                                         teff, gravb_bol, abun,
                                         irrad_frac_refl,
                                         mesh_method, is_single,
                                         do_rv_grav,
                                         features,
                                         **kwargs)

        # but we need to use the correct velocities for assigning RVs
        self.ind_self_vel = ind_self


    @classmethod
    def from_bundle(cls, b, component, compute=None,
                    datasets=[], pot=None, **kwargs):

        envelope = b.hierarchy.get_envelope_of(component)

        if pot is None:
            pot = b.get_value(qualifier='pot', component=envelope, context='component', **_skip_filter_checks)

        mesh_method_override = kwargs.pop('mesh_method', None)
        kwargs.setdefault('mesh_method', b.get_value(qualifier='mesh_method', component=envelope, compute=compute, mesh_method=mesh_method_override, **_skip_filter_checks) if compute is not None else 'marching')
        ntriangles_override = kwargs.pop('ntriangles', None)
        kwargs.setdefault('ntriangles', b.get_value(qualifier='ntriangles', component=envelope, compute=compute, ntriangles=ntriangles_override, **_skip_filter_checks) if compute is not None else 1000)

        return super(Star_roche_envelope_half, cls).from_bundle(b, component, compute,
                                                  datasets,
                                                  pot=pot,
                                                  **kwargs)


    @property
    def is_convex(self):
        return False

    @property
    def needs_remesh(self):
        """
        whether the star needs to be re-meshed (for any reason)
        """
        return False

    @property
    def _rpole_func(self):
        """
        """
        # the provided function must take *self.instantaneous_mesh_args as the
        # only arguments.  If this is not the case, the subclass must also override
        # instantaneous_rpole
        return getattr(libphoebe, 'roche_pole')

    @property
    def _gradOmega_func(self):
        """
        """
        # the provided function must take *self.instantaneous_mesh_args as the
        # only arguments.  If this is not the case, the subclass must also override
        # instantaneous_gpole
        return getattr(libphoebe, 'roche_gradOmega_only')

    @property
    def instantaneous_mesh_args(self):
        logger.debug("{}.instantaneous_mesh_args".format(self.component))

        if 'mesh_args' not in self.inst_vals.keys():
            logger.debug("{}.instantaneous_mesh_args COMPUTING".format(self.component))

            # self.q is automatically flipped to be 1./q for secondary components
            q = np.float64(self.q)

            F = np.float64(self.F)

            d = np.float64(self.instantaneous_d)

            Phi = self.pot

            self.inst_vals['mesh_args'] = q, F, d, Phi

        return self.inst_vals['mesh_args']

    def _build_mesh(self, mesh_method, **kwargs):
        """
        this function takes mesh_method and kwargs that came from the generic Body.intialize_mesh and returns
        the grid... intialize mesh then takes care of filling columns and rescaling to the correct units, etc
        """

        # need the sma to scale between Roche and real units
        sma = kwargs.get('sma', self.sma)  # Rsol (same units as coordinates)

        mesh_args = self.instantaneous_mesh_args

        if mesh_method == 'marching':
            # TODO: do this during mesh initialization only and then keep delta fixed in time??
            ntriangles = kwargs.get('ntriangles', self.ntriangles)

            # we need the surface area of the lobe to estimate the correct value
            # to pass for delta to marching.  We will later need the volume to
            # expose its value
            logger.debug("libphoebe.roche_area_volume{}".format(mesh_args))
            av = libphoebe.roche_area_volume(*mesh_args,
                                             choice=2,
                                             larea=True,
                                             lvolume=True)

            delta = _estimate_delta(ntriangles, av['larea'])

            logger.debug("libphoebe.roche_marching_mesh{}".format(mesh_args))
            try:
                new_mesh = libphoebe.roche_marching_mesh(*mesh_args,
                                                         delta=delta,
                                                         choice=2,
                                                         full=True,
                                                         max_triangles=int(ntriangles*1.5),
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
                                                         init_phi=kwargs.get('mesh_init_phi', self.mesh_init_phi))
            except Exception as err:
                if str(err) == 'There are too many triangles!':
                    mesh_init_phi_attempts = kwargs.get('mesh_init_phi_attempts', 1) + 1
                    if mesh_init_phi_attempts > 5:
                        raise err

                    mesh_init_phi = np.random.random()*2*np.pi
                    logger.warning("mesh failed to converge, trying attempt #{} with mesh_init_phi={}".format(mesh_init_phi_attempts, mesh_init_phi))
                    kwargs['mesh_init_phi_attempts'] = mesh_init_phi_attempts
                    kwargs['mesh_init_phi'] = mesh_init_phi
                    return self._build_mesh(mesh_method, **kwargs)
                else:
                    raise err

            # In addition to the values exposed by the mesh itself, let's report
            # the volume and surface area of the lobe.  The lobe area is used
            # if mesh_offseting is required, and the volume is optionally exposed
            # to the user.
            new_mesh['volume'] = av['lvolume']  # * sma**3
            new_mesh['area'] = av['larea']      # * sma**2

            scale = sma

        elif mesh_method == 'wd':
            N = int(kwargs.get('gridsize', self.gridsize))

            # unpack mesh_args
            q, F, d, Phi = mesh_args

            the_grid = mesh_wd.discretize_wd_style(N, q, F, d, Phi)
            new_mesh = mesh.wd_grid_to_mesh_dict(the_grid, q, F, d)
            scale = sma

        else:
            raise NotImplementedError("mesh_method '{}' is not supported".format(mesh_method))

        return new_mesh, scale


class Star_rotstar(Star):
    def __init__(self, component, comp_no, ind_self, ind_sibling,
                 masses, ecc, incl,
                 long_an, t0, do_mesh_offset, mesh_init_phi,

                 atm, datasets, passband, intens_weighting,
                 extinct, Rv,
                 ld_mode, ld_func, ld_coeffs, ld_coeffs_source,
                 lp_profile_rest,
                 requiv, sma,
                 polar_direction_uvw,
                 freq_rot,
                 teff, gravb_bol, abun,
                 irrad_frac_refl,
                 mesh_method, is_single,
                 do_rv_grav,
                 features,

                 **kwargs):
        """
        """
        # extra things (not used by Star) will be stored in kwargs

        super(Star_rotstar, self).__init__(component, comp_no, ind_self, ind_sibling,
                                           masses, ecc, incl,
                                           long_an, t0,
                                           do_mesh_offset, mesh_init_phi,

                                           atm, datasets, passband, intens_weighting,
                                           extinct, Rv,
                                           ld_mode, ld_func, ld_coeffs, ld_coeffs_source,
                                           lp_profile_rest,
                                           requiv, sma,
                                           polar_direction_uvw,
                                           freq_rot,
                                           teff, gravb_bol, abun,
                                           irrad_frac_refl,
                                           mesh_method, is_single,
                                           do_rv_grav,
                                           features,
                                           **kwargs)

    @classmethod
    def from_bundle(cls, b, component, compute=None,
                    datasets=[], **kwargs):


        return super(Star_rotstar, cls).from_bundle(b, component, compute,
                                                    datasets,
                                                    **kwargs)



    @property
    def is_convex(self):
        return True

    @property
    def needs_recompute_instantaneous(self):
        # recompute instantaneous for asynchronous spots, even if meshing
        # doesn't need to be recomputed
        return self.needs_remesh or (not self.is_single and len(self.features) and self.F != 1)

    @property
    def needs_remesh(self):
        """
        whether the star needs to be re-meshed (for any reason)
        """
        # TODO: or self.distortion_method != 'keplerian'?? If Nbody orbits can change freq_rot in time, then we need to remesh
        return self.is_misaligned

    @property
    def _rpole_func(self):
        """
        """
        # the provided function must take *self.instantaneous_mesh_args as the
        # only arguments.  If this is not the case, the subclass must also override
        # instantaneous_rpole
        return getattr(libphoebe, 'rotstar_misaligned_pole')

    @property
    def _gradOmega_func(self):
        """
        """
        # the provided function must take *self.instantaneous_mesh_args as the
        # only arguments.  If this is not the case, the subclass must also override
        # instantaneous_gpole
        return getattr(libphoebe, 'rotstar_misaligned_gradOmega_only')

    @property
    def instantaneous_mesh_args(self):
        logger.debug("{}.instantaneous_mesh_args".format(self.component))

        if 'mesh_args' not in self.inst_vals.keys():
            logger.debug("{}.instantaneous_mesh_args COMPUTING".format(self.component))

            # TODO: we need a different scale if self._is_single==True
            freq_rot = self.freq_rot
            omega = rotstar.rotfreq_to_omega(freq_rot, M_star = self.masses[self.ind_self], scale=self.sma, solar_units=True)

            # polar_direction_xyz is instantaneous based on current true_anom
            s = self.polar_direction_xyz

            # NOTE: if we ever want to break volume conservation in time,
            # get_target_volume will need to take time or true anomaly
            # TODO: not sure if scaled should be True or False here
            target_volume = self.get_target_volume(scaled=False)
            logger.debug("libphoebe.rotstar_misaligned_Omega_at_vol(vol={}, omega={}, s={})".format(target_volume, omega, s))
            Phi = libphoebe.rotstar_misaligned_Omega_at_vol(target_volume,
                                                            omega, s)

            self.inst_vals['mesh_args'] = omega, s, Phi

        return self.inst_vals['mesh_args']


    def _build_mesh(self, mesh_method, **kwargs):
        """
        this function takes mesh_method and kwargs that came from the generic Body.intialize_mesh and returns
        the grid... intialize mesh then takes care of filling columns and rescaling to the correct units, etc
        """

        # need the sma to scale between Roche and real units
        sma = kwargs.get('sma', self.sma)  # Rsol (same units as coordinates)

        mesh_args = self.instantaneous_mesh_args

        if mesh_method == 'marching':
            ntriangles = kwargs.get('ntriangles', self.ntriangles)

            av = libphoebe.rotstar_misaligned_area_volume(*mesh_args,
                                                          larea=True,
                                                          lvolume=True)

            delta = _estimate_delta(ntriangles, av['larea'])

            try:
                new_mesh = libphoebe.rotstar_misaligned_marching_mesh(*mesh_args,
                                                                      delta=delta,
                                                                      full=True,
                                                                      max_triangles=int(ntriangles*1.5),
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
                                                                      init_phi=kwargs.get('mesh_init_phi', self.mesh_init_phi))
            except Exception as err:
                if str(err) == 'There are too many triangles!':
                    mesh_init_phi_attempts = kwargs.get('mesh_init_phi_attempts', 1) + 1
                    if mesh_init_phi_attempts > 5:
                        raise err

                    mesh_init_phi = np.random.random()*2*np.pi
                    logger.warning("mesh failed to converge, trying attempt #{} with mesh_init_phi={}".format(mesh_init_phi_attempts, mesh_init_phi))
                    kwargs['mesh_init_phi_attempts'] = mesh_init_phi_attempts
                    kwargs['mesh_init_phi'] = mesh_init_phi
                    return self._build_mesh(mesh_method, **kwargs)
                else:
                    raise err


            # In addition to the values exposed by the mesh itself, let's report
            # the volume and surface area of the lobe.  The lobe area is used
            # if mesh_offseting is required, and the volume is optionally exposed
            # to the user.
            new_mesh['volume'] = av['lvolume']
            new_mesh['area'] = av['larea']

            scale = sma

        else:
            raise NotImplementedError("mesh_method '{}' is not supported".format(mesh_method))

        return new_mesh, scale


class Star_sphere(Star):
    def __init__(self, component, comp_no, ind_self, ind_sibling,
                 masses, ecc, incl,
                 long_an, t0, do_mesh_offset, mesh_init_phi,

                 atm, datasets, passband, intens_weighting,
                 extinct, Rv,
                 ld_mode, ld_func, ld_coeffs, ld_coeffs_source,
                 lp_profile_rest,
                 requiv, sma,
                 polar_direction_uvw,
                 freq_rot,
                 teff, gravb_bol, abun,
                 irrad_frac_refl,
                 mesh_method, is_single,
                 do_rv_grav,
                 features,

                 **kwargs):
        """
        """
        # extra things (not used by Star) will be stored in kwargs
        # NOTHING EXTRA FOR SPHERE AT THE MOMENT

        super(Star_sphere, self).__init__(component, comp_no, ind_self, ind_sibling,
                                          masses, ecc, incl,
                                          long_an, t0,
                                          do_mesh_offset, mesh_init_phi,

                                          atm, datasets, passband, intens_weighting,
                                          extinct, Rv,
                                          ld_mode, ld_func, ld_coeffs, ld_coeffs_source,
                                          lp_profile_rest,
                                          requiv, sma,
                                          polar_direction_uvw,
                                          freq_rot,
                                          teff, gravb_bol, abun,
                                          irrad_frac_refl,
                                          mesh_method, is_single,
                                          do_rv_grav,
                                          features,
                                          **kwargs)

    @classmethod
    def from_bundle(cls, b, component, compute=None,
                    datasets=[], **kwargs):

        self_ps = b.filter(component=component, context='component', **_skip_filter_checks)

        return super(Star_sphere, cls).from_bundle(b, component, compute,
                                                   datasets,
                                                   **kwargs)


    @property
    def is_convex(self):
        return True

    @property
    def needs_remesh(self):
        """
        whether the star needs to be re-meshed (for any reason)
        """
        return False

    @property
    def _rpole_func(self):
        """
        """
        # the provided function must take *self.instantaneous_mesh_args as the
        # only arguments.  If this is not the case, the subclass must also override
        # instantaneous_rpole
        return getattr(libphoebe, 'sphere_pole')

    @property
    def _gradOmega_func(self):
        """
        """
        # the provided function must take *self.instantaneous_mesh_args as the
        # only arguments.  If this is not the case, the subclass must also override
        # instantaneous_gpole
        return getattr(libphoebe, 'sphere_gradOmega_only')

    @property
    def instantaneous_mesh_args(self):
        logger.debug("{}.instantaneous_mesh_args".format(self.component))

        if 'mesh_args' not in self.inst_vals.keys():
            logger.debug("{}.instantaneous_mesh_args COMPUTING".format(self.component))

            # NOTE: if we ever want to break volume conservation in time,
            # get_target_volume will need to take time or true anomaly
            target_volume = self.get_target_volume()
            logger.debug("libphoebe.sphere_Omega_at_vol(vol={})".format(target_volume))
            Phi = libphoebe.sphere_Omega_at_vol(target_volume)

            self.inst_vals['mesh_args'] = (Phi,)

        return self.inst_vals['mesh_args']

    def _build_mesh(self, mesh_method, **kwargs):
        """
        this function takes mesh_method and kwargs that came from the generic Body.intialize_mesh and returns
        the grid... intialize mesh then takes care of filling columns and rescaling to the correct units, etc
        """

        # if we don't provide instantaneous masses or smas, then assume they are
        # not time dependent - in which case they were already stored in the init
        sma = kwargs.get('sma', self.sma)  # Rsol (same units as coordinates)

        mesh_args = self.instantaneous_mesh_args

        if mesh_method == 'marching':
            ntriangles = kwargs.get('ntriangles', self.ntriangles)

            av = libphoebe.sphere_area_volume(*mesh_args,
                                              larea=True,
                                              lvolume=True)

            delta = _estimate_delta(ntriangles, av['larea'])

            try:
                new_mesh = libphoebe.sphere_marching_mesh(*mesh_args,
                                                          delta=delta,
                                                          full=True,
                                                          max_triangles=int(ntriangles*1.5),
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
                                                          init_phi=kwargs.get('mesh_init_phi', self.mesh_init_phi))
            except Exception as err:
                if str(err) == 'There are too many triangles!':
                    mesh_init_phi_attempts = kwargs.get('mesh_init_phi_attempts', 1) + 1
                    if mesh_init_phi_attempts > 5:
                        raise err

                    mesh_init_phi = np.random.random()*2*np.pi
                    logger.warning("mesh failed to converge, trying attempt #{} with mesh_init_phi={}".format(mesh_init_phi_attempts, mesh_init_phi))
                    kwargs['mesh_init_phi_attempts'] = mesh_init_phi_attempts
                    kwargs['mesh_init_phi'] = mesh_init_phi
                    return self._build_mesh(mesh_method, **kwargs)
                else:
                    raise err

            # In addition to the values exposed by the mesh itself, let's report
            # the volume and surface area of the lobe.  The lobe area is used
            # if mesh_offseting is required, and the volume is optionally exposed
            # to the user.
            new_mesh['volume'] = av['lvolume']
            new_mesh['area'] = av['larea']

            scale = sma

        else:
            raise NotImplementedError("mesh_method '{}' is not supported".format(mesh_method))

        return new_mesh, scale


class Star_none(Star):
    """
    Override everything to do nothing... the Star just exists to be a mass
    for dynamics (and possibly distortion of other stars), but will not have
    any mesh or produce any observables
    """
    @property
    def is_convex(self):
        return True

    @property
    def needs_remesh(self):
        """
        whether the star needs to be re-meshed (for any reason)
        """
        return False

    @classmethod
    def _return_nones(*args, **kwargs):
        return 0.0

    @property
    def _rpole_func(self):
        return self._return_nones

    @property
    def _gradOmega_func(self):
        return self._return_nones

    @property
    def instantaneous_mesh_args(self):
        return None

    @property
    def instantaneous_maxr(self):
        return 0.0

    def _build_mesh(self, mesh_method, **kwargs):
        return {}, 0.0

    def _offset_mesh(self, new_mesh):
        return new_mesh

    def update_position(self, time,
                        xs, ys, zs, vxs, vys, vzs,
                        ethetas, elongans, eincls,
                        ds=None, Fs=None,
                        ignore_effects=False,
                        component_com_x=None,
                        **kwargs):

        # scaledprotomesh = mesh.ScaledProtoMesh(scale=1.0, **{})
        #
        # # TODO: get rid of this ugly _value stuff
        # pos = (_value(xs[self.ind_self]), _value(ys[self.ind_self]), _value(zs[self.ind_self]))
        # vel = (_value(vxs[self.ind_self_vel]), _value(vys[self.ind_self_vel]), _value(vzs[self.ind_self_vel]))
        # euler = (_value(ethetas[self.ind_self]), _value(elongans[self.ind_self]), _value(eincls[self.ind_self]))
        # euler_vel = (_value(ethetas[self.ind_self_vel]), _value(elongans[self.ind_self_vel]), _value(eincls[self.ind_self_vel]))
        #
        # self._mesh = mesh.Mesh.from_scaledproto(scaledprotomesh,
        #                                         pos, vel, euler, euler_vel,
        #                                         np.array([0,0,1]),
        #                                         component_com_x)

        self._mesh = None


    def get_pblum_scale(self, *args, **kwargs):
        return 1.0

    def set_pblum_scale(self, *args, **kwargs):
        return

    def compute_luminosity(self, *args, **kwargs):
        return 0.0

    def _populate_lp(self, dataset, **kwargs):
        return {}

    def _populate_rv(self, dataset, **kwargs):
        return {}

    def _populate_lc(self, dataset, ignore_effects=False, **kwargs):
        return {}


class Envelope(Body):
    def __init__(self, component, halves, pot, q,
                 mesh_method,
                 **kwargs):
        """
        """
        self.component = component
        self._halves = halves
        self._pot = pot
        self._q = q
        self.mesh_method = mesh_method

    @classmethod
    def from_bundle(cls, b, component, compute=None,
                    datasets=[], **kwargs):

        # self_ps = b.filter(component=component, context='component', check_visible=False)

        stars = b.hierarchy.get_siblings_of(component, kind='star')
        if not len(stars)==2:
            raise ValueError("hieararchy cannot find two stars in envelope")

        pot = b.get_value(qualifier='pot', component=component, context='component', **_skip_filter_checks)

        orbit = b.hierarchy.get_parent_of(component)
        q = b.get_value(qualifier='q', component=orbit, context='component', **_skip_filter_checks)

        mesh_method_override = kwargs.pop('mesh_method', None)
        mesh_method = b.get_value(qualifier='mesh_method', component=component, compute=compute, mesh_method=mesh_method_override, **_skip_filter_checks) if compute is not None else 'marching'

        if conf.devel:
            mesh_init_phi_override = kwargs.pop('mesh_init_phi', 0.0)
            try:
                mesh_init_phi = b.get_value(qualifier='mesh_init_phi', compute=compute, component=component, unit=u.rad, mesh_init_phi=mesh_init_phi_override, **_skip_filter_checks)
            except ValueError:
                kwargs.setdefault('mesh_init_phi', mesh_init_phi_override)
            else:
                kwargs.setdefault('mesh_init_phi', mesh_init_phi)
        else:
            kwargs.setdefault('mesh_init_phi', 0.0)

        # we'll pass on the potential from the envelope to both halves (even
        # though technically only the primary will ever actually build a mesh)
        halves = [Star_roche_envelope_half.from_bundle(b, star, compute=compute, datasets=datasets, pot=pot, mesh_method=mesh_method, **kwargs) for star in stars]

        return cls(component, halves, pot, q, mesh_method)

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, system):
        self._system = system
        for half in self._halves:
            half.system = system

    @property
    def boosting_method(self):
        return self._boosting_method

    @boosting_method.setter
    def boosting_method(self, boosting_method):
        self._boosting_method = boosting_method
        for half in self._halves:
            half.boosting_method = boosting_method

    @property
    def halves(self):
        return {half.component: half for half in self._halves}

    def get_half(self, component):
        return self.halves[component]

    @property
    def meshes(self):
        # TODO: need to combine self._halves[0].mesh and self._halves[1].mesh and handle indices, volume?
        return mesh.Meshes(self.halves)

    @property
    def mesh(self):
        return self.meshes

    def update_position(self, *args, **kwargs):
        def split_mesh(mesh, q, pot):
            logger.debug("splitting envelope mesh according to neck min")

            # compute position of nekmin (d=1.)
            logger.debug("split_mesh libphoebe.roche_contact_neck_min(q={}, d={}, pot={})".format(q, 1., pot))
            nekmin = libphoebe.roche_contact_neck_min(np.pi / 2., q, 1., pot)['xmin']

            # initialize the subcomp array
            subcomp = np.zeros(len(mesh['triangles']))
            # default value is 0 for primary, need to set 1 for secondary
            subcomp[mesh['centers'][:, 0] > nekmin] = 1

            # will need to catch all vertices that are on the wrong side of the center
            # get x coordinates of vertices per triangle, subtract nekmin to evaluate the side they're on
            xs_vert_triang = mesh['vertices'][:, 0][mesh['triangles']] - nekmin
            # assign 0 for primary and 1 for secondary
            xs_vert_triang[xs_vert_triang < 0] = 0
            xs_vert_triang[xs_vert_triang > 0] = 1

            env_comp_verts = np.zeros(len(mesh['vertices']))
            env_comp_triangles = np.zeros(len(mesh['triangles']))

            env_comp_verts[mesh['vertices'][:,0] > nekmin] = 1
            env_comp_triangles[mesh['centers'][:,0] > nekmin] = 1

            # summing comp values per triangle flags those with mismatching vertex and triangle comps

            vert_comp_triang = np.sum(xs_vert_triang, axis=1)
            # vert_comp_triang = 0/3 - all on primary/secondary
            # vert_comp_triang = 1 - two vertices on primary, one on secondary
            # vert_comp_triang = 2 - one vertex on primary, two on secondary

            # find indices of triangles with boundary crossing vertices

            triangind_primsec = np.argwhere(((vert_comp_triang == 1) | (vert_comp_triang == 2)) & (subcomp == 0)).flatten()
            triangind_secprim = np.argwhere(((vert_comp_triang == 1) | (vert_comp_triang == 2)) & (subcomp == 1)).flatten()

            # to get the indices of the vertices that need to be copied because they cross from prim to sec:
            vertind_primsec = mesh['triangles'][triangind_primsec][xs_vert_triang[triangind_primsec] == 1]
            # and sec to prim:
            vertind_secprim = mesh['triangles'][triangind_secprim][xs_vert_triang[triangind_secprim] == 0]

            # combine the two in an array for convenient stacking of copied vertices
            vinds_tocopy = np.hstack((vertind_primsec,vertind_secprim))

            # this one can be merged into less steps
            # vertices_primcopy = np.vstack((mesh['vertices'], mesh['vertices'][vertind_primsec]))
            # vertices_seccopy = np.vstack((vertices_primcopy, mesh['vertices'][vertind_secprim]))
            new_triangle_indices_prim = range(len(mesh['vertices']), len(mesh['vertices'])+len(vertind_primsec))
            new_triangle_indices_sec = range(len(mesh['vertices'])+len(vertind_primsec), len(mesh['vertices'])+len(vertind_primsec)+len(vertind_secprim))

            mesh['vertices'] = np.vstack((mesh['vertices'], mesh['vertices'][vinds_tocopy]))
            mesh['pvertices'] = np.vstack((mesh['pvertices'], mesh['pvertices'][vinds_tocopy]))
            mesh['vnormals'] = np.vstack((mesh['vnormals'], mesh['vnormals'][vinds_tocopy]))
            mesh['normgrads'] = np.hstack((mesh['normgrads'].vertices, mesh['normgrads'].vertices[vinds_tocopy]))
            mesh['velocities'] = np.vstack((mesh['velocities'].vertices, np.zeros((len(vinds_tocopy),3))))
            env_comp_verts = np.hstack((env_comp_verts, env_comp_verts[vinds_tocopy]))

            # change the env_comp value of the copied vertices (hopefully right?)
            env_comp_verts[new_triangle_indices_prim] = 0
            env_comp_verts[new_triangle_indices_sec] = 1

            # the indices of the vertices in the triangles array (crossing condition) need to be replaced with the new ones
            # a bit of array reshaping magic, but it works
            triangind_primsec_f = mesh['triangles'][triangind_primsec].flatten().copy()
            triangind_secprim_f = mesh['triangles'][triangind_secprim].flatten().copy()
            indices_prim = np.where(np.in1d(triangind_primsec_f, vertind_primsec))[0]
            indices_sec = np.where(np.in1d(triangind_secprim_f, vertind_secprim))[0]

            triangind_primsec_f[indices_prim] = new_triangle_indices_prim
            triangind_secprim_f[indices_sec] = new_triangle_indices_sec

            mesh['triangles'][triangind_primsec] = triangind_primsec_f.reshape(len(triangind_primsec_f) // 3, 3)
            mesh['triangles'][triangind_secprim] = triangind_secprim_f.reshape(len(triangind_secprim_f) // 3, 3)

            # NOTE: this doesn't update the stored entries for scalars (volume, area, etc)
            mesh_halves = [mesh.take(env_comp_triangles==0, env_comp_verts==0), mesh.take(env_comp_triangles==1, env_comp_verts==1)]

            # we now need to recompute the areas and volumes of each half separately
            # nekmin = libphoebe.roche_contact_neck_min(np.pi/2., q, 1.0, pot)['xmin']
            # for compno,mesh in enumerate(mesh_halves):
                # component passed here is expected to be 1 or 2 (not 0 or 1)
                # info0 = libphoebe.roche_contact_partial_area_volume(nekmin, q, 1.0, pot, compno+1)
                # mesh._volume = info0['lvolume']
                # mesh._area = info0['lvolume']

            return mesh_halves

        if not (self._halves[0].has_standard_mesh() and self._halves[1].has_standard_mesh()):
            # update the position (and build the mesh) of the primary component
            # this will internally call save_as_standard mesh with the mesh
            # of the ENTIRE contact envelope.
            self._halves[0].update_position(*args, **kwargs)

            # now let's access this saved WHOLE mesh
            mesh_contact = self._halves[0].get_standard_mesh(scaled=False)

            # and split it according to the x-position of neck min
            mesh_primary, mesh_secondary = split_mesh(mesh_contact, self._q, self._pot)

            # now override the standard mesh with just the corresponding halves
            self._halves[0].save_as_standard_mesh(mesh_primary)
            self._halves[1].save_as_standard_mesh(mesh_secondary)


        # since the standard mesh already exists, this should simply handle
        # placing in orbit.  We'll do this again for the primary so it
        # will update to just the correct half.  This is a bit redundant,
        # but keeps all this logic out of the Star classes
        for half, com in zip(self._halves, [0, 1]):
            half.update_position(component_com_x=com, *args, **kwargs)

    def compute_luminosity(self, *args, **kwargs):
        return np.sum([half.compute_luminosity(*args, **kwargs) for half in self._halves])

    def set_pblum_scale(self, *args, **kwargs):
        # allow backends to attempt to set the scale for the envelope, but ignore
        # as halves will each have their own scaling
        return

    def populate_observable(self, time, kind, dataset, **kwargs):
        """
        TODO: add documentation
        """

        for half in self._halves:
            half.populate_observable(time, kind, dataset, **kwargs)


################################################################################
################################################################################
################################################################################


class Feature(object):
    """
    Note that for all features, each of the methods below will be called.  So
    changing the coordinates WILL affect the original/intrinsic loggs which
    will then be used as input for that method call.

    In other words, its probably safest if each feature only overrides a
    SINGLE one of the methods.  Overriding multiple methods should be done
    with great care.

    Each feature may or may not require recomputing a mesh, depending on the
    kind of change it exacts to the mesh. For example, pulsations will require
    recomputing a mesh while spots will not. By default, the mesh will be
    recomputed (set in this superclass' `__init__()` method) but inherited
    classes should overload `self._remeshing_required`.
    """
    def __init__(self, *args, **kwargs):
        pass

    @property
    def _remeshing_required(self):
        return True

    @property
    def proto_coords(self):
        """
        Override this to True if all methods (except process_coords*... those
        ALWAYS expect protomesh coordinates) are expecting coordinates
        in the protomesh (star) frame-of-reference rather than the
        current in-orbit system frame-of-reference.
        """
        return False

    def process_coords_for_computations(self, coords_for_computations, s, t):
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

    def process_coords_for_observations(self, coords_for_computations, coords_for_observations, s, t):
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

    def process_loggs(self, loggs, coords, s=np.array([0., 0., 1.]), t=None):
        """
        Method for a feature to process the loggs.

        Features that affect loggs should override this method
        """
        return loggs

    def process_teffs(self, teffs, coords, s=np.array([0., 0., 1.]), t=None):
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

        feature_ps = b.get_feature(feature=feature, **_skip_filter_checks)

        colat = feature_ps.get_value(qualifier='colat', unit=u.rad, **_skip_filter_checks)
        longitude = feature_ps.get_value(qualifier='long', unit=u.rad, **_skip_filter_checks)

        if len(b.hierarchy.get_stars())>=2:
            star_ps = b.get_component(component=feature_ps.component, **_skip_filter_checks)
            orbit_ps = b.get_component(component=b.hierarchy.get_parent_of(feature_ps.component), **_skip_filter_checks)
            # TODO: how should this handle dpdt?

            # we won't use syncpar directly because that is defined wrt sidereal period and we want to make sure
            # this translated to roche longitude correctly.  In the non-apsidal motion case
            # syncpar = period_anom_orb / period_star
            period_anom_orb = orbit_ps.get_value(qualifier='period_anom', unit=u.d, **_skip_filter_checks)
            period_star = star_ps.get_value(qualifier='period', unit=u.d, **_skip_filter_checks)
            dlongdt = 2*pi * (period_anom_orb/period_star - 1) / period_anom_orb
        else:
            star_ps = b.get_component(component=feature_ps.component, **_skip_filter_checks)
            dlongdt = star_ps.get_value(qualifier='freq', unit=u.rad/u.d, **_skip_filter_checks)
            longitude += np.pi/2

        radius = feature_ps.get_value(qualifier='radius', unit=u.rad, **_skip_filter_checks)
        relteff = feature_ps.get_value(qualifier='relteff', unit=u.dimensionless_unscaled, **_skip_filter_checks)

        t0 = b.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)

        return cls(colat, longitude, dlongdt, radius, relteff, t0)

    @property
    def _remeshing_required(self):
        return False

    @property
    def proto_coords(self):
        """
        """
        return True

    def pointing_vector(self, s, time):
        """
        s is the spin vector in roche coordinates
        time is the current time
        """
        t = time - self._t0
        longitude = self._longitude + self._dlongdt * t

        # define the basis vectors in the spin (primed) coordinates in terms of
        # the Roche coordinates.
        # ez' = s
        # ex' =  (ex - s(s.ex)) /|i - s(s.ex)|
        # ey' = s x ex'
        ex = np.array([1., 0., 0.])
        ezp = s
        exp = (ex - s*np.dot(s,ex))
        eyp = np.cross(s, exp)

        return np.sin(self._colat)*np.cos(longitude)*exp +\
                  np.sin(self._colat)*np.sin(longitude)*eyp +\
                  np.cos(self._colat)*ezp

    def process_teffs(self, teffs, coords, s=np.array([0., 0., 1.]), t=None):
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

        pointing_vector = self.pointing_vector(s,t)
        logger.debug("spot.process_teffs at t={} with pointing_vector={} and radius={}".format(t, pointing_vector, self._radius))

        cos_alpha_coords = np.dot(coords, pointing_vector) / np.linalg.norm(coords, axis=1)
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

        feature_ps = b.get_feature(feature=feature, **_skip_filter_checks)
        freq = feature_ps.get_value(qualifier='freq', unit=u.d**-1, **_skip_filter_checks)
        radamp = feature_ps.get_value(qualifier='radamp', unit=u.dimensionless_unscaled, **_skip_filter_checks)
        l = feature_ps.get_value(qualifier='l', unit=u.dimensionless_unscaled, **_skip_filter_checks)
        m = feature_ps.get_value(qualifier='m', unit=u.dimensionless_unscaled, **_skip_filter_checks)
        teffext = feature_ps.get_value(qualifier='teffext', **_skip_filter_checks)

        GM = c.G.to('solRad3 / (solMass d2)').value*b.get_value(qualifier='mass', component=feature_ps.component, context='component', unit=u.solMass, **_skip_filter_checks)
        R = b.get_value(qualifier='rpole', component=feature_ps.component, section='component', unit=u.solRad, **_skip_filter_checks)

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

    def process_coords_for_computations(self, coords_for_computations, s, t):
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

    def process_coords_for_observations(self, coords_for_computations, coords_for_observations, s, t):
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

    def process_teffs(self, teffs, coords, s=np.array([0., 0., 1.]), t=None):
        """
        """
        if not self._teffext:
            return teffs

        raise NotImplementedError("teffext=True not yet supported for pulsations")
