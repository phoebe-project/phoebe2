import numpy as np
import astropy.units as u

import phoebe.parameters.feature as _parameters_feature

import logging
logger = logging.getLogger("COMPONENT_FEATURES")
logger.addHandler(logging.NullHandler())


__all__ = ['register_feature', 'ComponentFeature', 'Spot', 'Pulsation']

_skip_filter_checks = {'check_default': False, 'check_visible': False}

def register_feature(feature_cls, kind=None):
    if kind is None:
        kind = feature_cls.__name__.lower()

    _parameters_feature._register(feature_cls, kind)
    globals()[kind.title()] = feature_cls
    __all__.append(kind.title())

class ComponentFeature(object):
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
    classes should overload `self.remeshing_required`.

    remeshing_required:

    proto_coords: Override this to True if all methods (except modify_coords*... those
        ALWAYS expect protomesh coordinates) are expecting coordinates
        in the protomesh (star) frame-of-reference rather than the
        current in-orbit system frame-of-reference.
    """
    _phoebe_custom_feature = 'dataset'
    allowed_component_kinds = ['star', 'envelope', 'orbit']
    allowed_dataset_kinds = [None]
    remeshing_required = True
    proto_coords = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def from_bundle(cls, b, feature):
        return cls(**cls.parse_bundle(b, feature))

    @classmethod
    def parse_bundle(cls, b, feature):
        return {}

    @classmethod
    def get_parameters(self, feature, **kwargs):
        raise NotImplementedError("get_parameters must be implemented in the feature subclass")

    def modify_coords_for_computations(self, coords_for_computations, s, t):
        """
        Method for a feature to modify the coordinates.  Coordinates are
        modified AFTER scaling but BEFORE being placed in orbit.

        NOTE: coords_for_computations affect physical properties only and
        not geometric properties (areas, eclipse detection, etc).  If you
        want to override geometric properties, use the hook for
        modify_coords_for_observations as well.

        Features that affect coordinates_for_computations should override
        this method
        """
        return coords_for_computations

    def modify_coords_for_observations(self, coords_for_computations, coords_for_observations, s, t):
        """
        Method for a feature to modify the coordinates.  Coordinates are
        modified AFTER scaling but BEFORE being placed in orbit.

        NOTE: coords_for_observations affect the geometry only (areas of each
        element and eclipse detection) but WILL NOT affect any physical
        parameters (loggs, teffs, intensities).  If you want to override
        physical parameters, use the hook for modify_coords_for_computations
        as well.

        Features that affect coordinates_for_observations should override this method.
        """
        return coords_for_observations

    def modify_loggs(self, loggs, coords, s=[0., 0., 1.], t=None):
        """
        Method for a feature to modify the loggs.

        Features that affect loggs should override this method
        """
        return loggs

    def modify_teffs(self, teffs, coords, s=[0., 0., 1.], t=None):
        """
        Method for a feature to modify the teffs.

        Features that affect teffs should override this method
        """
        return teffs

    def modify_intensities(self, abs_normal_intensities, normal_intensities, abs_intensities, intensities,
                            coords, s=[0., 0., 1.], t=None):
        """
        Method for a feature to modify the intensities.
        Features that affect intensities should override this method
        """
        return abs_normal_intensities, normal_intensities, abs_intensities, intensities

class Spot(ComponentFeature):
    remeshing_required = False
    proto_coords = True

    @classmethod
    def parse_bundle(cls, b, feature):
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
            dlongdt = 2 * np.pi * (period_anom_orb/period_star - 1) / period_anom_orb
        else:
            star_ps = b.get_component(component=feature_ps.component, **_skip_filter_checks)
            dlongdt = star_ps.get_value(qualifier='freq', unit=u.rad/u.d, **_skip_filter_checks)
            longitude += np.pi/2

        radius = feature_ps.get_value(qualifier='radius', unit=u.rad, **_skip_filter_checks)
        relteff = feature_ps.get_value(qualifier='relteff', unit=u.dimensionless_unscaled, **_skip_filter_checks)

        t0 = b.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)

        return dict(colat=colat, longitude=longitude, dlongdt=dlongdt, radius=radius, relteff=relteff, t0=t0)

    def pointing_vector(self, s, time):
        """
        s is the spin vector in roche coordinates
        time is the current time
        """
        t = time - self.kwargs['t0']
        longitude = self.kwargs['longitude'] + self.kwargs['dlongdt'] * t
        colat = self.kwargs['colat']

        # define the basis vectors in the spin (primed) coordinates in terms of
        # the Roche coordinates.
        # ez' = s
        # ex' =  (ex - s(s.ex)) /|i - s(s.ex)|
        # ey' = s x ex'
        ex = np.array([1., 0., 0.])
        ezp = s
        exp = (ex - s*np.dot(s,ex))
        eyp = np.cross(s, exp)

        return np.sin(colat)*np.cos(longitude)*exp +\
                  np.sin(colat)*np.sin(longitude)*eyp +\
                  np.cos(colat)*ezp

    def modify_teffs(self, teffs, coords, s=np.array([0., 0., 1.]), t=None):
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
        logger.debug("spot.modify_teffs at t={} with pointing_vector={} and radius={}".format(t, pointing_vector, self.kwargs['radius']))

        cos_alpha_coords = np.dot(coords, pointing_vector) / np.linalg.norm(coords, axis=1)
        cos_alpha_spot = np.cos(self.kwargs['radius'])

        filter_ = cos_alpha_coords > cos_alpha_spot
        teffs[filter_] = teffs[filter_] * self.kwargs['relteff']

        return teffs

class Pulsation(ComponentFeature):
    proto_coords = True

    @classmethod
    def parse_bundle(cls, b, feature):
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

        return dict(radamp=radamp, freq=freq, l=l, m=m, tanamp=tanamp, teffext=teffext)

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

    def modify_coords_for_computations(self, coords_for_computations, s, t):
        """
        """
        if self.kwargs['teffext']:
            return coords_for_computations

        x, y, z, r = coords_for_computations[:,0], coords_for_computations[:,1], coords_for_computations[:,2], np.sqrt((coords_for_computations**2).sum(axis=1))
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)

        xi_r = self.kwargs['radamp'] * Y(self.kwargs['m'], self.kwargs['l'], theta, phi) * np.exp(-1j*2*np.pi*self.kwargs['freq']*t)
        xi_t = self.kwargs['tanamp'] * self.dYdtheta(self.kwargs['m'], self.kwargs['l'], theta, phi) * np.exp(-1j*2*np.pi*self.kwargs['freq']*t)
        xi_p = self.kwargs['tanamp']/np.sin(theta) * self.dYdphi(self.kwargs['m'], self.kwargs['l'], theta, phi) * np.exp(-1j*2*np.pi*self.kwargs['freq']*t)

        new_coords = np.zeros(coords_for_computations.shape)
        new_coords[:,0] = coords_for_computations[:,0] + xi_r * np.sin(theta) * np.cos(phi)
        new_coords[:,1] = coords_for_computations[:,1] + xi_r * np.sin(theta) * np.sin(phi)
        new_coords[:,2] = coords_for_computations[:,2] + xi_r * np.cos(theta)

        return new_coords

    def modify_coords_for_observations(self, coords_for_computations, coords_for_observations, s, t):
        """
        Displacement equations:

          xi_r(r, theta, phi)     = a(r) Y_lm (theta, phi) exp(-i*2*pi*f*t)
          xi_theta(r, theta, phi) = b(r) dY_lm/dtheta (theta, phi) exp(-i*2*pi*f*t)
          xi_phi(r, theta, phi)   = b(r)/sin(theta) dY_lm/dphi (theta, phi) exp(-i*2*pi*f*t)

        where:

          b(r) = a(r) GM/(R^3*f^2)
        """
        # TODO: we do want to displace the coords_for_observations, but the x,y,z,r below are from the ALSO displaced coords_for_computations
        # if not self.kwargs['teffext']:
            # return coords_for_observations

        x, y, z, r = coords_for_computations[:,0], coords_for_computations[:,1], coords_for_computations[:,2], np.sqrt((coords_for_computations**2).sum(axis=1))
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)

        xi_r = self.kwargs['radamp'] * Y(self.kwargs['m'], self.kwargs['l'], theta, phi) * np.exp(-1j*2*np.pi*self.kwargs['freq']*t)
        xi_t = self.kwargs['tanamp'] * self.dYdtheta(self.kwargs['m'], self.kwargs['l'], theta, phi) * np.exp(-1j*2*np.pi*self.kwargs['freq']*t)
        xi_p = self.kwargs['tanamp']/np.sin(theta) * self.dYdphi(self.kwargs['m'], self.kwargs['l'], theta, phi) * np.exp(-1j*2*np.pi*self.kwargs['freq']*t)

        new_coords = np.zeros(coords_for_observations.shape)
        new_coords[:,0] = coords_for_observations[:,0] + xi_r * np.sin(theta) * np.cos(phi)
        new_coords[:,1] = coords_for_observations[:,1] + xi_r * np.sin(theta) * np.sin(phi)
        new_coords[:,2] = coords_for_observations[:,2] + xi_r * np.cos(theta)

        return new_coords

    def modify_teffs(self, teffs, coords, s=np.array([0., 0., 1.]), t=None):
        """
        """
        if not self.kwargs['teffext']:
            return teffs

        raise NotImplementedError("teffext=True not yet supported for pulsations")
