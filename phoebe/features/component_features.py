import numpy as np
import astropy.units as u

from phoebe.parameters import FloatParameter, ParameterSet
from phoebe.features.common import BaseFeature


__all__ = ['ComponentFeature', 'Spot', 'Pulsation']

_skip_filter_checks = {'check_default': False, 'check_visible': False}


class ComponentFeature(BaseFeature):
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
    classes should overload `self.requires_remeshing`.
    """
    allowed_component_kinds = ['star', 'envelope']
    allowed_dataset_kinds = [None]

    def __repr__(self):
        return f"<ComponentFeature: {self.__class__.__name__}>"

    def requires_remeshing(self):
        """
        Whether this feature requires remeshing of the component mesh.
        """
        return False

    def cartesian_to_spherical(self, roche_coords):
        """
        Transform Cartesian Roche coordinates to spherical coordinates.
        
        Parameters
        ----------
        roche_coords : array_like
            Array of Cartesian coordinates with shape (N, 3) where columns
            are [x, y, z]
            
        Returns
        -------
        r : ndarray
            Radial distance from origin
        theta : ndarray
            Colatitude (polar angle) in radians, measured from positive
            z-axis [0, π]
        phi : ndarray
            Longitude (azimuthal angle) in radians, measured from positive
            x-axis [-π, π]
        """
        import numpy as np
        
        x, y, z = roche_coords[:, 0], roche_coords[:, 1], roche_coords[:, 2]
        r = np.sqrt((roche_coords**2).sum(axis=1))
        theta = np.arccos(z/r)  # colatitude [0, π]
        phi = np.arctan2(y, x)  # longitude [-π, π]
        
        return r, theta, phi

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

    def modify_rvs(self, rvs, orbit_vel, roche_coords, s=[0., 0., 1.], t=None):
        """
        Method for a feature to modify the radial velocities.

        Features that affect radial velocities (RV+LP datasets) should override this method

        NOTE: orbit_vel[2] is in the OPPOSITE direction of the radial velocity
        """
        return rvs

    def modify_loggs(self, loggs, roche_coords, s=[0., 0., 1.], t=None):
        """
        Method for a feature to modify the loggs.

        Features that affect loggs should override this method
        """
        return loggs

    def modify_teffs(self, teffs, roche_coords, s=[0., 0., 1.], t=None):
        """
        Method for a feature to modify the teffs.

        Features that affect teffs should override this method
        """
        return teffs

    def modify_intensities(self, abs_normal_intensities, abs_intensities,
                           mus, pblum_scale, extinct_factors, boost_factors,
                           roche_coords, s=[0., 0., 1.], t=None):
        """
        Method for a feature to modify the intensities.
        Features that affect intensities should override this method

        Arguments
        ----------
        * `abs_normal_intensities` (ndarray): Absolute normal intensities, already multiplied
            by `extinct_factors`.
        * `abs_intensities` (ndarray): Absolute projected intensities, already multiplied by
            `extinct_factors` and `boost_factors`.
        * `mus` (ndarray): Cosine of the angle between the normal vector and the line of sight
        * `pblum_scale` (ndarray): Scale factor for the pblum, that will be applied to the abs_intensities
            AFTER modify_intensities to result in the scaled intensities
        * `extinct_factors` (ndarray): Extinction factors for the intensities, already applied
        * `boost_factors` (ndarray): Boost factors for the intensities, already applied
        * `roche_coords` (ndarray): Roche coordinates for the computations
        * `s` (array-like): Spin vector in Roche coordinates
        * `t` (float): Current time
        """
        return abs_normal_intensities, abs_intensities


class Spot(ComponentFeature):
    @classmethod
    def create_feature_parameters(cls, feature, **kwargs):
        """
        Create a <phoebe.parameters.ParameterSet> for a spot feature.

        Generally, this will be used as an input to the kind argument in
        <phoebe.frontend.bundle.Bundle.add_feature>.  If attaching through
        <phoebe.frontend.bundle.Bundle.add_feature>, all `**kwargs` will be
        passed on to set the values as described in the arguments below.  Alternatively,
        see <phoebe.parameters.ParameterSet.set_value> to set/change the values
        after creating the Parameters.

        Allowed to attach to:
        * components with kind: star
        * datasets: not allowed

        Arguments
        ----------
        * `colat` (float/quantity, optional): colatitude of the center of the spot
            wrt spin axis.
        * `long` (float/quantity, optional): longitude of the center of the spot wrt
            spin axis.
        * `radius` (float/quantity, optional): angular radius of the spot.
        * `relteff` (float/quantity, optional): temperature of the spot relative
            to the intrinsic temperature.

        Returns
        --------
        * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
            <phoebe.parameters.Parameter> objects and a list of all necessary
            constraints.
        """
        params = []
        params += [FloatParameter(qualifier="colat", value=kwargs.get('colat', 0.0), default_unit=u.deg, description='Colatitude of the center of the spot wrt spin axis')]
        params += [FloatParameter(qualifier="long", value=kwargs.get('long', 0.0), default_unit=u.deg, description='Longitude of the center of the spot wrt spin axis')]
        params += [FloatParameter(qualifier='radius', value=kwargs.get('radius', 1.0), limits=(0, 180), default_unit=u.deg, description='Angular radius of the spot')]
        params += [FloatParameter(qualifier='relteff', value=kwargs.get('relteff', 1.0), limits=(0, None), default_unit=u.dimensionless_unscaled, description='Temperature of the spot relative to the intrinsic temperature')]

        return ParameterSet(params), []

    @classmethod
    def parse_bundle(cls, b, feature_ps):
        """
        Initialize a Spot feature from the bundle.
        """
        import numpy as np
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
            rot_dlongdt = 2 * np.pi * (period_anom_orb/period_star - 1) / period_anom_orb
        else:
            star_ps = b.get_component(component=feature_ps.component, **_skip_filter_checks)
            rot_dlongdt = star_ps.get_value(qualifier='freq', unit=u.rad/u.d, **_skip_filter_checks)
            longitude += np.pi/2

        radius = feature_ps.get_value(qualifier='radius', unit=u.rad, **_skip_filter_checks)
        relteff = feature_ps.get_value(qualifier='relteff', unit=u.dimensionless_unscaled, **_skip_filter_checks)

        t0 = b.get_value(qualifier='t0', context='system', unit=u.d, **_skip_filter_checks)

        return dict(colat=colat, longitude=longitude, rot_dlongdt=rot_dlongdt, radius=radius, relteff=relteff, t0=t0)

    @classmethod
    def run_checks_compute(cls, b, feature_ps, compute_ps):
        items = []
        relteff_param = feature_ps.get_parameter(qualifier='relteff', **_skip_filter_checks)
        radius_param = feature_ps.get_parameter(qualifier='radius', **_skip_filter_checks)
        if relteff_param.get_value(**_skip_filter_checks) == 1:
            items += [{'msg': 'relteff of spot is 1.0 which will have no affect',
                       'params': [relteff_param],
                       'is_error': False}]
        if radius_param.get_value(**_skip_filter_checks) == 0:
            items += [{'msg': 'radius of spot is 0.0 which will have no affect',
                       'params': [radius_param],
                       'is_error': False}]
        return items

    def requires_remeshing(self):
        return True

    def instantaneous_position(self, s, time):
        """
        s is the spin vector in roche coordinates
        time is the current time
        """
        t = time - self.kwargs['t0']
        longitude = self.kwargs['longitude'] + self.kwargs['rot_dlongdt'] * t
        colat = self.kwargs['colat']
        return longitude, colat

    def pointing_vector(self, s, time):
        """
        s is the spin vector in roche coordinates
        time is the current time
        """
        import numpy as np
        longitude, colat = self.instantaneous_position(s, time)

        """
        define the basis vectors in the spin (primed) coordinates in terms of
        the Roche coordinates.
        - the z' direction is explicitly given by the spin vector
        ez' = s
        - the x' direction should point in the longitudinal direction that the 
        companion is in the rotating frame, which means that the y' direction should 
        always be orthogonal to x. Thus, to get this, we'll first calculate the y' 
        direction by taking the cross product of the z' and x directions 
        ey' = ez' x ex
        if the pole aligns with the x-axis, then the y' direction is the z direction
        - the x' direction is then defined by the cross product of the other two
        ex' = ey' x ez'
        """
        ex = np.array([1., 0., 0.])
        ezp = s / np.linalg.norm(s)
        if (s == ex).all():
            eyp = np.array([0., 0., 1.])
        else:
            eyp = np.cross(ezp, ex)
        exp = np.cross(eyp, ezp)

        # now we can express the pointing vector in terms of the primed basis
        pv = (np.sin(colat)*np.cos(longitude)*exp +
              np.sin(colat)*np.sin(longitude)*eyp +
              np.cos(colat)*ezp)

        # renormalize and return pointing vector
        return pv / np.linalg.norm(pv)

    def modify_teffs(self, teffs, roche_coords, s=[0., 0., 1.], t=None):
        """
        Change the local effective temperatures for any values within the
        "cone" defined by the spot.  Any teff within the spot will have its
        current value multiplied by the "relteff" factor

        :parameter array teffs: array of teffs for computations
        :parameter array coords: array of coords for computations
        :t float: current time
        """
        import numpy as np
        s = np.asarray(s)
        if t is None:
            # then assume at t0
            t = self.kwargs['t0']

        pointing_vector = self.pointing_vector(s, t)
        self.logger.debug("spot.modify_teffs at t={} with pointing_vector={} and radius={}".format(t, pointing_vector, self.kwargs['radius']))

        cos_alpha_coords = np.dot(roche_coords, pointing_vector) / np.linalg.norm(roche_coords, axis=1)
        cos_alpha_spot = np.cos(self.kwargs['radius'])

        filter_ = cos_alpha_coords > cos_alpha_spot
        teffs[filter_] = teffs[filter_] * self.kwargs['relteff']

        return teffs


class Pulsation(ComponentFeature):
    @classmethod
    def parse_bundle(cls, b, feature_ps):
        """
        Initialize a Pulsation feature from the bundle.
        """
        from phoebe import c
        freq = feature_ps.get_value(qualifier='freq', unit=u.d**-1, **_skip_filter_checks)
        radamp = feature_ps.get_value(qualifier='radamp', unit=u.dimensionless_unscaled, **_skip_filter_checks)
        l = feature_ps.get_value(qualifier='l', unit=u.dimensionless_unscaled, **_skip_filter_checks)
        m = feature_ps.get_value(qualifier='m', unit=u.dimensionless_unscaled, **_skip_filter_checks)
        teffext = feature_ps.get_value(qualifier='teffext', **_skip_filter_checks)

        GM = c.G.to('solRad3 / (solMass d2)').value*b.get_value(qualifier='mass', component=feature_ps.component, context='component', unit=u.solMass, **_skip_filter_checks)
        R = b.get_value(qualifier='rpole', component=feature_ps.component, section='component', unit=u.solRad, **_skip_filter_checks)

        tanamp = GM/R**3/freq**2

        return dict(radamp=radamp, freq=freq, l=l, m=m, tanamp=tanamp, teffext=teffext)

    @classmethod
    def Y(self, m, l, theta, phi):
        try:
            from scipy.special import sph_harm as Y
        except ImportError:
            from scipy.special import sph_harm_y as _sph_harm_y

        def Y(m, n, theta, phi):
            return _sph_harm_y(n, m, theta, phi)

        return Y(m, l, theta, phi)

    def dYdtheta(self, m, l, theta, phi):
        if abs(m) > l:
            return 0

        # TODO: just a quick hack
        if abs(m+1) > l:
            last_term = 0.0
        else:
            last_term = self.Y(m+1, l, theta, phi)

        return m/np.tan(theta)*self.Y(m, l, theta, phi) + np.sqrt((l-m)*(l+m+1))*np.exp(-1j*phi)*last_term

    def dYdphi(self, m, l, theta, phi):
        return 1j*m*self.Y(m, l, theta, phi)

    def modify_coords_for_computations(self, coords_for_computations, s, t):
        """
        """
        if self.kwargs['teffext']:
            return coords_for_computations

        x, y, z, r = coords_for_computations[:,0], coords_for_computations[:,1], coords_for_computations[:,2], np.sqrt((coords_for_computations**2).sum(axis=1))
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)

        xi_r = self.kwargs['radamp'] * self.Y(self.kwargs['m'], self.kwargs['l'], theta, phi) * np.exp(-1j*2*np.pi*self.kwargs['freq']*t)
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

        xi_r = self.kwargs['radamp'] * self.Y(self.kwargs['m'], self.kwargs['l'], theta, phi) * np.exp(-1j*2*np.pi*self.kwargs['freq']*t)
        xi_t = self.kwargs['tanamp'] * self.dYdtheta(self.kwargs['m'], self.kwargs['l'], theta, phi) * np.exp(-1j*2*np.pi*self.kwargs['freq']*t)
        xi_p = self.kwargs['tanamp']/np.sin(theta) * self.dYdphi(self.kwargs['m'], self.kwargs['l'], theta, phi) * np.exp(-1j*2*np.pi*self.kwargs['freq']*t)

        new_coords = np.zeros(coords_for_observations.shape)
        new_coords[:,0] = coords_for_observations[:,0] + xi_r * np.sin(theta) * np.cos(phi)
        new_coords[:,1] = coords_for_observations[:,1] + xi_r * np.sin(theta) * np.sin(phi)
        new_coords[:,2] = coords_for_observations[:,2] + xi_r * np.cos(theta)

        return new_coords

    def modify_teffs(self, teffs, roche_coords, s=np.array([0., 0., 1.]), t=None):
        """
        """
        if not self.kwargs['teffext']:
            return teffs

        raise NotImplementedError("teffext=True not yet supported for pulsations")
