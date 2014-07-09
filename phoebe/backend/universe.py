"""
Classes representing meshes and functions to handle them.

**Base classes**

.. autosummary::
    
   Body
   PhysicalBody
   BodyBag
   BinaryBag

**Specific Bodies**
   
.. autosummary::    

   Star
   BinaryStar
   BinaryRocheStar
   AccretionDisk
   
**Helper functions**

.. autosummary::    

   get_binary_orbit
   luminosity
   load
   
**For developers**

.. autosummary::    

   keep_only_results
   merge_results
   init_mesh
   compute_scale_or_offset
    
Section 1. Basic meshing
========================

Subsection 1.1. Body
--------------------

A mesh is represented by a L{Body}. This class takes care of the position of the
L{Body} in the Universe, it's rotation (L{Body.rotate}), translation
(L{Body.translate}), and detection of visibility of triangles
(L{Body.detect_eclipse_horizon}). There are no constraints on the morphology of
the mesh, as long as it is closed. I.e., there can be holes, it can contain
multiple detached bodies etc... but it cannot be a "sheet". It must be closed
because each triangle only has one normal vector. For example, a mesh cannot
consist of a single triangle.

All classes representing a body in the universe should subclass L{Body}.

The minimal set of information to generate is mesh is the explicit definition of
all the triangles. It is then possible to compute the normals, sizes etc... of
the triangles numerically. Alternatively (and recommended), one can also compute
these quantities from implicit equations. See L{Body} for more details and
examples.

.. inheritance-diagram:: Body

Subsection 1.2. PhysicalBody
----------------------------

A L{PhysicalBody} inherits all capabilities of a L{Body}. It extends these
capabilities by making subdivision available, and implementing functions to
generate light curves (L{PhysicalBody.lc}), radial velocity curves
(L{PhysicalBody.rv}), spectra (L{PhysicalBody.spectrum}) and interferometric
visibilities (L{PhysicalBody.ifm}).

.. inheritance-diagram:: PhysicalBody

Subsection 1.3. BodyBag
-----------------------

One can easily collect different Bodies in one container, i.e. the L{BodyBag}.
The BodyBag has all functionality of a Body, but not those of a L{PhysicalBody}.
All functions implemented in L{Body} are applied to the merged mesh of all
Bodies within the L{BodyBag} (e.g., rotation). This means that you cannot make
BodyBag out of Bodies with different labels in their C{mesh}. If a function is
called that is not implemented in L{Body} or L{BodyBag}, the L{BodyBag} will
assume that the function exists for every L{Body} in the L{BodyBag}, and call
these functions. E.g. calling an intensity function on the BodyBag, will set
the intensities for all Bodies individually.

.. inheritance-diagram:: BodyBag

.. inheritance-diagram:: BinaryBag

Subsection 1.4. Subclassing PhysicalBody
----------------------------------------

Specific meshes, i.e. meshes that can easily be parametrized, can be represented
by a class that subclasses the L{PhysicalBody} class for convenience. These kind
of classes facilitate the computation of the mesh. A simple example is
L{Ellipsoid}, parametrized by the radius of the ellipsoid in the x, y and z
direction. One can also add functionality to these classes, which will be

.. inheritance-diagram:: Star

.. inheritance-diagram:: BinaryRocheStar

.. inheritance-diagram:: BinaryStar


exploited in the next Section.

Section 2. Physical meshing
===========================

Some meshes represent a physical entity, e.g. a star. These classes (subclassing
L{PhysicalBody}) add functionality and properties to the Body. These include
local effective temperatures, surface gravities, intensities, velocities etc...
Most of the physical meshes are dependent on time, so they implement a method
L{set_time} which takes care of setting all the properties of the mesh on that
time point. A typical extra functionality of the physical meshes is that they
have intensities, and thus they also implement a method L{projected_intensity}
which sets the local projected intensities, but also sums them all up to get the
total observed flux.

Section 3. Computing observational quantities
=============================================

Section 4. Input and Output
===========================

Subsection 4.1 Datafiles
------------------------

You can save Bodies and their results via L{Body.save}. This creates a binary
file (pickled). To load and thus restore a Body, call L{load}.

Subsection 4.2 Plots
--------------------

Make 3D plots with L{Body.plot3D} or make 2D plots with L{Body.plot2D}. The
latter is just a shortcut to L{observatory.image}.

"""
enable_mayavi = True
# Load standard libraries
import pickle
import uuid
import logging
import copy
import textwrap
import types
from collections import OrderedDict
# Load 3rd party modules
import numpy as np
from numpy import sin,cos,pi,sqrt,pi
from scipy.integrate import quad
from scipy.optimize import nnls, fmin
import scipy
try:
    import pylab as pl
except ImportError:
    pass
if enable_mayavi:
    try:
        from enthought.mayavi import mlab
    except:
        try:
            from mayavi import mlab
        except:
            enable_mayavi = False
from phoebe.units import conversions
from phoebe.units import constants
from phoebe.utils import coordinates
from phoebe.utils import utils
from phoebe.utils import fgeometry
from phoebe.utils.decorators import memoized, clear_memoization
try:
    from phoebe.utils import cgeometry
except ImportError:
    pass
from phoebe.algorithms import marching
from phoebe.algorithms import subdivision
from phoebe.algorithms import eclipse
from phoebe.algorithms import interp_nDgrid
from phoebe.algorithms import reflection
from phoebe.backend import decorators
from phoebe.backend import observatory
from phoebe.backend import processing
from phoebe.parameters import parameters
from phoebe.parameters import datasets
from phoebe.parameters import tools
from phoebe.atmospheres import roche
from phoebe.atmospheres import limbdark
from phoebe.atmospheres import spots
from phoebe.atmospheres import pulsations
from phoebe.atmospheres import magfield
from phoebe.atmospheres import velofield
from phoebe.atmospheres import reddening
from phoebe.dynamics import keplerorbit
try:
    from phoebe.utils import transit
except ImportError:
    pass

# We are not interested in messages from numpy, but we are in messages from our
# own code: that's why we create a logger.
np.seterr(all='ignore')
logger = logging.getLogger("UNIVERSE")
logger.addHandler(logging.NullHandler())

# some easy conversions
rsol_2_au = constants.Rsol/constants.au
kms_2_rsold = 1000. / constants.Rsol * 24.0 * 3600.0
m_2_au = 1.0/constants.au
deg_2_rad = pi / 180.
day_2_sec = 86400.0
Nld_law = 5 # number of limb darkening coefficients + 1

#{ Functions of general interest    

def get_binary_orbit(self, time):
    """
    Get the orbital coordinates of a Body (and it's companion) at a certain time.
    
    This function is a small wrapper around the function
    :py:func:`get_orbit <phoebe.dynamics.keplerorbit.get_orbit>` from the
    ``keplerorbit`` module inside the ``dynamics`` package.
    
    @param self: a Body
    @type self: Body
    @param time: the time in the orbit
    @type time: float
    @return: location and velocity of primar, secondary and distance between components
    @rtype: [x1, y1, z1, vx1, vy1, vz1], [x2, y2, z2, vx2, vy2, vz2], d
    """
    # Get some information on the orbit
    P = self.params['orbit']['period']
    e = self.params['orbit']['ecc']
    a = self.params['orbit']['sma'] * constants.Rsol
    q = self.params['orbit']['q']
    a1 = a / (1 + 1.0/q)
    a2 = a - a1
    inclin = self.params['orbit']['incl'] * deg_2_rad
    argper = self.params['orbit']['per0'] * deg_2_rad
    long_an = self.params['orbit']['long_an'] * deg_2_rad
    T0 = self.params['orbit']['t0']
    t0type = self.params['orbit']['t0type']
    
    if t0type == 'superior conjunction':
        time = time - self.params['orbit']['phshift'] * P
    
    # Where in the orbit are we?
    time *= day_2_sec
    P *= day_2_sec
    T0 *= day_2_sec
    loc1, velo1, euler1 = keplerorbit.get_orbit(time, P, e, a1, T0, per0=argper,
                                      long_an=long_an, incl=inclin,
                                      component='primary', t0type=t0type)
    loc2, velo2, euler2 = keplerorbit.get_orbit(time, P, e, a2, T0, per0=argper,
                                      long_an=long_an, incl=inclin,
                                      component='secondary', t0type=t0type)
    
    # We need everything in cartesian Rsol units
    loc1 = np.array(loc1) / a
    loc2 = np.array(loc2) / a
    
    # Compute the distance between the two components
    d = sqrt( (loc1[0]-loc2[0])**2 + \
              (loc1[1]-loc2[1])**2 + \
              (loc1[2]-loc2[2])**2)
    
    # That's it, thank you for your attention
    return list(loc1) + list(velo1), list(loc2) + list(velo1), d

    
def luminosity(body, ref='__bol', numerical=False):
    r"""
    Calculate the total luminosity of an object.
    
    It integrates the limbdarkening law over the solid angle on one hemisphere
    of the triangle, for every triangle on the surface:
    
    .. math::
    
       L = 2\pi\int_S \int_{0}^{\frac{\pi}{2}}I_\mathrm{bol}(\cos\theta)\cos\theta\sin\theta d\theta dA \quad\mathrm{[erg/s]}
    
    A dimensional analysis confirms the units: :math:`I_\mathrm{bol}` is the bolometric specific intensity,
    so this is :math:`\mathrm{erg}/\mathrm{s}/\mathrm{cm}^2/\AA/\mathrm{sr}` integrated over wavelength, and thus
    :math:`\mathrm{erg}/\mathrm{s}/\mathrm{cm}^2/\mathrm{sr}`.
    Next, a solid angle integration is performed, removing
    the sterradian (:math:`\mathrm{erg}/\mathrm{s}/\mathrm{cm}^2`). Finally, an integration over the surface
    removes the last units and the result is erg/s.
    
    Here, :math:`\theta` is the angle between the line of sight and the surface
    normal, i.e. :math:`\theta=0` at the center of the disk.
    
    @return: luminosity of the object (W)
    @rtype: float
    """
    parset, ref = body.get_parset(ref=ref)
    # Set the intensities if they are not calculated yet
    ld_law = parset['ld_func']
    ld = body.mesh['ld_' + ref]
    
    if np.all(ld==0):
        body.intensity(ref=ref, beaming_alg='none')
        
    # Get a reference to the mesh, and get the sizes of the triangles in real
    # units
    mesh = body.mesh
    sizes = mesh['size'] * constants.Rsol**2
    
    
    if numerical:
        # Get the function to evaluate the LD law
        ld_law = getattr(limbdark, 'ld_'+ld_law)
        # Define the function to compute the total intrinsic emergent flux
        def _tief(gamma, coeffs):
            """Small helper function to compute total intrinsic emergent flux"""
            cos_gamma = cos(gamma)
            Imu = coeffs[-1] * ld_law(cos_gamma, coeffs)
            # sin(gamma) is for solid angle integration
            return Imu * cos_gamma * sin(gamma)
    
        # Then do integration:
        emer_Ibolmu = 2*pi*np.array([quad(_tief, 0, pi/2, args=(ld[i],))[0] \
                                                     for i in range(len(mesh))])
    else:
        # Get the function to evaluate the LD law
        ld_disk = getattr(limbdark, 'disk_'+ld_law)
        emer_Ibolmu = ld_disk(ld[:,:-1].T) * ld[:,-1]
        
    return (emer_Ibolmu * sizes).sum()
    

def generic_projected_intensity(system, los=[0.,0.,+1], method='numerical',
                beaming_alg='none', ld_func='claret', ref=0,
                with_partial_as_half=True):
    r"""
    Calculate local projected intensity.
    
    We can speed this up if we compute the local intensity first, keep track of
    the limb darkening coefficients and evaluate for different angles. Then we
    only have to do a table lookup once.
    
    Analytical calculation can also be an approximation!
    
    Other things that are done when computing projected intensities:
    
        1. Correction for distance to the source
        2. Correction for interstellar reddening if the passband is not bolometric
        3. Scattering phase functions
        
    Additionally, we can take **beaming** into account if we're using the linear
    approximation (:envvar:`beaming_alg='local'` or :envvar:`beaming_alg='simple'`).
    In that case the (local) intensity :math:`I` is adjusted to :math:`I_b` to
    include beaming effects with a beaming amplitude :math:`A_b` as follows:
    
    .. math::
    
        I_b = A_b I
    
    with
    
    .. math::
    
        A_b = 1 + \alpha_b \frac{v(z)}{c}
        
    where :math:v(z) is the radial velocity of the surface element and :math:`\alpha_b`
    is the beaming factor, computed in :py:func:`compute_grid_ld_coeffs <phoebe.atmospheres.create_atmospherefits.compute_grid_ld_coeffs>` (see the link for more info):
    
    .. math::
            
        \alpha_b = \frac{\int_P (5+\frac{d\ln F_\lambda}{d\ln\lambda}\lambda F_\lambda d\lambda}{\int_P \lambda F_\lambda d\lambda}
    
    @param system: object to compute temperature of
    @type system: Body or derivative class
    @param los: line-of-sight vector. Best leave it at the default
    @type los: list of three floats
    @param method: flag denoting type of calculation: numerical or analytical approximation
    @type method: str ('numerical' or 'analytical')
    @param ld_func: limb-darkening model
    @type ld_func: str
    @param ref: ref of self's observation set to compute local intensities of
    @type ref: str
    """
    
    # Retrieve some basic information on the body, passband set, third light
    # and passband luminosity
    body = system.params.values()[0]
    idep, ref = system.get_parset(ref=ref, type='pbdep')
    
    # Numerical computation (as opposed to analytical)
    if method == 'numerical':
        
        # Get limb angles
        mus = system.mesh['mu']
        # To calculate the total projected intensity, we keep track of the
        # partially visible triangles, and the totally visible triangles
        
        # (the correction for nans in the size of the faces is probably due to a
        # bug in the marching method, though I'm not sure):
        #if np.any(np.isnan(system.mesh['size'])):
        #    print('encountered nan')
        #    raise SystemExit
        
        keep = (mus>0) & (system.mesh['partial'] | system.mesh['visible'])
        
        # Create shortcut variables
        vis_mesh = system.mesh[keep]
        mus = mus[keep]
        
        # Negating the next array gives the partially visible things, that is
        # the only reason for defining it.
        visible = vis_mesh['visible']
        partial = vis_mesh['partial']
        
        # Compute intensity using the already calculated limb darkening
        # coefficents, or interpolate here if we're using the Prsa method.
        logger.debug('using limbdarkening law {}'.format((ld_func)))
        if ld_func != 'prsa':
            Imu = getattr(limbdark, 'ld_{}'.format(ld_func))(mus, vis_mesh['ld_'+ref].T)\
                      * vis_mesh['ld_'+ref][:,-1]
        else:
            Imu = limbdark.ld_intensity_prsa(system, idep)
        
        
        # Do the beaming correction
        if beaming_alg == 'simple' or beaming_alg == 'local':
            # Retrieve beming factor
            alpha_b = vis_mesh['alpha_b_' + ref]
            # light speed in Rsol/d
            ampl_b = 1.0 + alpha_b * vis_mesh['velo___bol_'][:,2]/37241.94167601236
        else:
            ampl_b = 1.0
                  
        proj_Imu = ampl_b * mus * Imu
        if with_partial_as_half:
            proj_Imu[partial] /= 2.0 
        
        # Take care of reflected light
        if ('refl_'+ref) in system.mesh.dtype.names:
            # Anisotropic scattering is difficult, we need to figure out where
            # the companion is. Note that this doesn't really work for multiple
            # systems; we should add different refl columns for that so that we
            # can track the origin of the irradiation.
            if 'scattering' in idep and idep['scattering'] != 'isotropic':
                
                # First, figure out where the companion is.
                sibling = system.get_sibling()
                
                X2 = (sibling.mesh['center']*sibling.mesh['size'][:,None]/sibling.mesh['size'].sum()).sum(axis=0)
                
                # What are the lines-of-sight between this companion and its
                # sibling?
                los = X2[None,:] - vis_mesh['center']
                
                # Then, what is are the angles between the line-of-sights to the
                # triangles, and the los from the triangle to the sibling?
                mu = coordinates.cos_angle(los, np.array([[0,0,1]]), axis=1)
                
                # Henyey-Greenstein phase function
                if idep['scattering'] == 'henyey':
                    g = idep['asymmetry']
                    #phase_function = 0.5 * (1-g**2) / (1 + g**2 - 2*g*mu)**1.5
                    phase_function = reflection.henyey_greenstein(mu, g)
                
                # Henyey-Greenstein phase function
                elif idep['scattering'] == 'henyey2':
                    g1 = idep['asymmetry1']
                    g2 = idep['asymmetry2']
                    f = idep['scat_ratio']
                    phase_function = reflection.henyey_greenstein2(mu, g1, g2, f)
                
                # Rayleigh phase function
                elif idep['scattering'] == 'rayleigh':
                    phase_function = reflection.rayleigh(mu)
                    
                # Hapke model (http://stratus.ssec.wisc.edu/streamer/userman/surfalb.html)
                # Vegetation (clover): w=0.101, q=-0.263, S=0.589, h=0.046 (Pinty and Verstraete 1991)
                # Snow: w=0.99, q=0.6, S=0.0, h=0.995 (Domingue 1997, Verbiscer and Veverka 1990)
                elif idep['scattering'] == 'hapke':
                    
                    # single scattering albedo
                    w = idep['alb']
                    Q = idep['asymmetry']
                    S = idep['hot_spot_ampl']
                    h = idep['hot_spot_width']
                    
                    # cosine of incidence angle
                    mup = coordinates.cos_angle(los, vis_mesh['normal_'], axis=1)
                    
                    # Phase function
                    phase_function = reflection.hapke(mu, mup, mus, w, Q, S, h)
                
                proj_Imu += vis_mesh['refl_'+ref] * mus * phase_function / np.pi
                logger.info("Projected intensity contains reflected light with phase function {}".format(idep['scattering']))
                
                
            # Isotropic scattering is easy, we don't need to figure out where
            # the companion is
            else:            
                
                proj_Imu += vis_mesh['refl_'+ref] * mus / np.pi 
                logger.info("Projected intensity contains isotropic reflected light")
            
        proj_intens = vis_mesh['size']*proj_Imu
        
        # Fill in the mesh
        system.mesh['proj_'+ref] = 0.
        system.mesh['proj_'+ref][keep] = proj_Imu
    
    
    # Analytical computation (as opposed to numerical)
    elif method == 'analytical':
        lcdep, ref = system.get_parset(ref)
        # The projected intensity is normalised with the distance in cm, we need
        # to reconvert that into solar radii.
        proj_intens = limbdark.sphere_intensity(body,lcdep)[1]/(constants.Rsol)**2
        
        
    # Scale the projected intensity with the distance
    globals_parset = system.get_globals()
    if globals_parset is not None:
        # distance in solar radii
        distance = globals_parset.get_value('distance') * 3.085677581503e+16 / constants.Rsol
    else:
        distance = 1.0 
    
    proj_intens = proj_intens.sum()/distance**2    
    
    
    # Take reddening into account (if given), but only for non-bolometric
    # passbands and nonzero extinction
    if ref != '__bol':
        
        # if there is a global reddening law
        red_parset = system.get_globals('reddening')
        if (red_parset is not None) and (red_parset['extinction'] > 0):
            ebv = red_parset['extinction'] / red_parset['Rv']
            proj_intens = reddening.redden(proj_intens,
                         passbands=[idep['passband']], ebv=ebv, rtype='flux',
                         law=red_parset['law'])[0]
            logger.info("Projected intensity is reddened with E(B-V)={} following {}".format(ebv, red_parset['law']))
        
        # if there is passband reddening
        elif 'extinction' in idep and (idep['extinction'] > 0):
            extinction = idep['extinction']
            proj_intens = proj_intens / 10**(extinction/2.5)
            logger.info("Projected intensity is reddened with extinction={} (passband reddening)".format(extinction))
    
    
    # That's all folks!
    return proj_intens
    
    
def load(filename):
    """
    Load a class from a file.
    
    Any class defined in this module can be loaded.
    
    @param filename: location of the file
    @type filename: str
    @return: Body saved in file
    @rtype: Body
    """
    ff = open(filename, 'r')
    myclass = pickle.load(ff)
    ff.close()
    return myclass


def keep_only_results(system):
    """
    Remove all information from this Body except for the results.
    
    The results are then still located in the C{params} attribute, but all other
    parameterSets and meshes are removed.
    
    It can be handy to remove unnecessary information from a Body before passing
    it around via the MPI protocol.
    """
    if hasattr(system, 'params'):
        for key in system.params:
            if not key == 'syn':
                trash = system.params.pop(key)
                del trash
    if hasattr(system, 'parent'):
        system.parent = None
    if hasattr(system, 'subdivision'):
        system.subdivision = {}
    if hasattr(system, 'bodies'):
        for body in system.bodies:
            keep_only_results(body)
    system.remove_mesh()
    return system


def remove_disabled_data(system):
    """
    Remove disabled data.
    
    This can ben handy to remove unncessary data from a Body before passing it
    around via the MPI protocol
    """
    if hasattr(system, 'params'):
        if 'obs' in system.params:
            for obstype in system.params['obs']:
                for iobs in system.params['obs'][obstype]:
                    if not system.params['obs'][obstype][iobs].get_enabled():
                        thrash = system.params['obs'][obstype].pop(iobs)
                        del thrash
    if hasattr(system, 'bodies'):
        for body in system.bodies:
            remove_disabled_data(body)
    return system
    


def merge_results(list_of_bodies):
    """
    Merge results of a list of bodies.
    
    Evidently, the bodies need to be representing the same system.
    
    It is of vital importance that each system in the list has the exact
    same hierarchy of results, as they will be iterated over simultaneously.
    It is not important how many or what the ``data`` and ``pbdep`` values are.
    
    The body that is returned is actually the first body in the list.
    
    Could be useful for MPI stuff.
    
    @param list_of_bodies: a list of bodies for which the results need to be
     merged
    @type list_of_bodies: list of Bodies
    @return: the first body but with the results of all the others added
    @rtype: Body
    """
    # Walk through all results of all bodies simultaneously
    iterators = [body.walk_type(type='syn') for body in list_of_bodies]
    for iteration in zip(*iterators):
        for ps in iteration[1:]:
            # Skip stuff that is not a result, but this should already be
            # the case
            if ps.context[-3:] == 'syn':
                # We want to merge all lists in the parametersets that are
                # in the 'columns' parameter.
                for key in ps:
                    if 'columns' in ps and not key in ps['columns']:
                        continue
                    value = ps[key]
                    # Again, this if statement is probably redundant
                    if isinstance(value,list):
                        iteration[0][key] += value
    return list_of_bodies[0]

   
def init_mesh(self):
    """
    Initialize a mesh.
    
    This adds columns to the mesh record array according to the dependables, and
    resets any existing mesh to all zeros.
    
    @param self: the Physical body to set the mesh of
    @type self: PhysicalBody
    """
    # Wrap everything up in one array, but first see how many lds columns we
    # need: for sure the bolometric one, but for the rest, this is dependent on
    # on the pbdep parameters (note that at this point, we just prepare the
    # array, we don't do anything with it yet):
    
    # We need the length of the current mesh, and note that our limb-darkening
    # laws can have maximum 5 parameters (4 + intensity)
    N = len(self.mesh)
    
    # We construct all the fields that are needed in this Body. Then we check
    # which ones already exist, and remove those
        
    # Bolometric intensities (we don't need 'bolometric' velocities because
    # they have been added by the base class (Body) already
    lds = [('ld___bol', 'f8', (Nld_law,)), ('proj___bol', 'f8')]
    
    # Velocities and passband intensities. Actually we don't need the
    # velocities. The bolometric velocities should be combined with the
    # passband luminosities to  compute the passband velocities. I still
    # retain them here because I don't know if the code will crash if I
    # remove them. That can be tested once we have an extensive automatic
    # test suite
    for pbdeptype in self.params['pbdep']:
        for iobs in self.params['pbdep'][pbdeptype]:
            iobs = self.params['pbdep'][pbdeptype][iobs]
            lds.append(('ld_{0}'.format(iobs['ref']), 'f8', (Nld_law,)))
            lds.append(('proj_{0}'.format(iobs['ref']), 'f8'))
            #lds.append(('velo_{0}_'.format(iobs['ref']), 'f8', (3,)))
            #lds.append(('_o_velo_{0}_'.format(iobs['ref']), 'f8', (3,)))
    
    # Basic fields
    lds = lds + [('logg','f8'), ('teff','f8'), ('abun','f8')]
    
    # Remove the ones that already exist:
    lds = [ild for ild in lds if not ild[0] in self.mesh.dtype.names]
    
    # Basic info
    dtypes = np.dtype(self.mesh.dtype.descr + lds)
        
    # Add a magnetic field if necessary
    if 'magnetic_field' in self.params and not 'B_' in dtypes.names:
        dtypes = np.dtype(dtypes.descr + \
                 [('B_', 'f8', (3,)), ('_o_B_','f8', (3,))])
    
    # for compatibility when loading pickled old style bundles (before July 2014)
    # (you can remove the following line if we're sure they are not going to be
    # used anymore):
    if not hasattr(self, '_extra_mesh_columns'):
        self._extra_mesh_columns = {}
    
    # Add any user mesh columns
    dtypes = np.dtype(dtypes.descr + [col for col in self._extra_mesh_columns if not col[0] in self.mesh.dtype.names])
    
    self.mesh = np.zeros(N, dtype=dtypes)
    # We need to make sure to reset the body, otherwise we could be fooled
    # into thinking that everything is still calculated! Some bodies do not
    # recalculate anything when the time is already set (because they are
    # time independent). This function effectively puts all values in the
    # columns to zero!
    self.reset()

def check_input_ps(self, ps, contexts, narg, is_list=False):
    """
    Check if a given parameterSet is of a certain context, and if not, raise
    an error.
    """
    if ps is None:
        return None
    
    if isinstance(narg, int):
        if narg%10 == 1:
            name = '{}st'.format(narg)
        elif narg%10 == 2:
            name = '{}nd'.format(narg)
        elif narg%10 == 3:
            name = '{}rd'.format(narg)
        else:
            name = '{}th'.format(narg)
    else:
        name = "'"+narg+"'"
    
    if is_list:
        extra1 = 'list of '
        extra2 = 's'
        extra3 = 'found at least one'
    else:
        extra1 = ''
        extra2 = ''
        extra3 = 'not'
        
    # Check if it is a ParameterSet
    if not isinstance(ps, parameters.ParameterSet):
        if len(contexts)==1:
            context_msg = "of context '{}'".format(contexts[0])
        elif len(contexts)==2:
            context_msg = "of context '{}' or '{}'".format(contexts[0], contexts[1])
        else:
            context_msg = "of context '{}' or '{}'".format("', '".join(contexts[:-1]), contexts[-1])
        raise ValueError(("{} argument in {} should be a {}ParameterSet{} {}, "
                          "{} '{}'").format(name, self.__class__.__name__,
                          extra1, extra2, context_msg, extra3,
                          type(ps).__name__))
    
    # Check if it is of the right context
    if not ps.get_context() in contexts:
        if len(contexts)==1:
            context_msg = "'{}'".format(contexts[0])
        elif len(contexts)==2:
            context_msg = "'{}' or '{}'".format(contexts[0], contexts[1])
        else:
            context_msg = "any of '{}' or '{}'".format("', '".join(contexts[:-1]), contexts[-1])
        raise ValueError(("Context{} of {} argument in {} should be {}"
                          "{}, {} '{}'").format(extra2, name, 
                          self.__class__.__name__, extra1, context_msg, extra3,
                          ps.get_context()))



def compute_scale_or_offset(model, obs, sigma=None, scale=False, offset=False,
                        type='nnls'):
    r"""
    Rescale the observations to match a model.
    
    .. math::
        
        O = \mathtt{scale} * S + \mathtt{offset}
    
    where :math:`O` represents the observations, :math:`S` the synthetics (the
    model), :math:`\mathtt{scale}` represents a linear scaling factor and
    :math:`\mathtt{offset}` represents an offset term. The parameter
    :envvar:`sigma` represent the uncertainty on the
    observations, which are used as weights in the fit. If they are not given,
    they are set to unity.
    
    The units of ``offset`` are the observer's units: i.e. if the observations
    are normalised to 1, ``offset`` will be fractional. If they are normalised
    to 100, ``offset`` is in percentage. If they are in absolute flux units,
    then also ``offset`` is in absolute flux units.
    
    The user has the choice to fit none of, only one of, or both of ``scale``
    and ``offset``.
    
    Note that the philosophy of this function is that we do **not** (I repeat
    do **not**) touch the observations in **any** circumstance. Our model should
    generate the data, not the other way around. Of course sometimes you want to
    process the observations before; you can, but do so before passing them to
    Phoebe (e.g. normalizing a spectrum). Parameters that you want to introduce
    (either fixed or to be fitted) should be stored inside the ``obs`` DataSets,
    and the data generating functions (``lc`` etc..) should know about them.
    
    Fitting types:
        - :envvar:`type='nnls'` does not allow negative coefficients: this can
          be useful for light curves: you perhaps don't want to accommodate for
          negative third light. Equivalently, this prohibits you from enhancing
          contrasts or amplitudes, but only allows them to be decreased.
        - :envvar:`type='lstsq'` does allow negative coefficients: this can be
          used for spectral lines: if the model profile is shallower than the
          observations, you can still enlargen them.
          
    The type only has effect if ``scale`` and ``offset`` need to be fitted
    simultaneously. If only ``scale`` is to be fitted, the scaling factor
    is computed as
    
    .. math::
    
        \mathtt{scale} = \frac{\sum_i w_i \frac{O_i}{S_i}}{\sum_i w_i},
        
    with
    
    .. math::
    
        w_i = \left(\frac{\sigma_i}{S_i}\right)^{-2}.
        
    If only the offset needs to be computed, it is computed as
    
    .. math::
    
        \mathtt{offset} = \frac{\sum_i w_i (O_i-S_i)}{\sum_i w_i},
        
    with
    
    .. math::
    
        w_i = \sigma_i^{-2}.
        
    """
    # Choose the algorithm
    algorithm = dict(nnls=nnls, lstsq=np.linalg.lstsq)[type]
    
    if np.isscalar(sigma):
        sigma = sigma*np.ones_like(obs)
    elif sigma is None or not len(sigma):
        sigma = np.ones_like(obs)
    
    #   only scaling factor
    if scale and not offset:
        scale = np.average(obs / model, weights=model**2/sigma**2)
    
    #   only offset
    elif not scale and offset:
        offset = np.average(obs - model, weights=1.0/sigma**2)
    
    #   scaling factor and offset
    elif scale and offset:
        #~ print model.ravel().shape, obs.ravel().shape
        A = np.column_stack([model.ravel(), np.ones(len(model.ravel()))])
        #~ print A.shape
        scale, offset = algorithm(A, obs.ravel())[0]
    
    return scale, offset



                    
                    
def _parse_pbdeps(body, pbdep, take_defaults=None):
    """
    Attach passband dependables to a body.
    
    This function takes care of separating different types of dependables,
    and attaching them in the C{params} dictionary, an attribute of a
    L{PhysicalBody}. Observables are for example parameterSets of type C{lcdep},
    C{rvdep} or C{spdep} (non-exhaustive list).
    
    First, this function checks if dependables are actually given. That is, it
    cannot be equal to C{None}.
    
    Next, the function checks whether only one single dependable has been given.
    Since we actually expect a list but know what to do with a single dependable
    too, we simply put the single dependable in a list.
    
    If C{body} has no C{pbdep} entry yet in the C{body.params} dictionary, a new
    (ordered) dictionary will be created.
    
    Finally, it checks what types of dependables are given, and each of them
    will be added to the ordered dictionary of the dependable. The key of each
    dependable is its reference (ref).
    
    For each added pbdep, also a "syn" equivalent will be created for
    convenience. It is possible that it stays empty during the course of the
    computations, but that's a problem for other functions. We know nothing
    here, we're from Barcelona.
    
    Working with an ordered dictionary separated according to dependable type
    enables us to unambiguously reference a certain dependable set with a
    reference (duh), but also with an index (aha!). E.g. the first C{lcdep} set
    that is added is referenced by index number 0. This is handy because if you
    only work with one or two light curves, the user shouldn't be bothered
    with thinkig about names for them (but can still do so if he/she wishes so).
    
    The function returns a list of all the references of the dependables that
    have been parsed.
    
    @param body: the body to attach the dependables to
    @type body: Body
    @param pbdep: a list of ParameterSets containing dependables
    @type pbdep: list
    @return: list of parsed references
    @rtype: list of str
    """
    # Map pbdeps to DataSets
    result_sets = dict(lcdep=datasets.LCDataSet,
                       rvdep=datasets.RVDataSet,
                       spdep=datasets.SPDataSet,
                       pldep=datasets.PLDataSet,
                       ifdep=datasets.IFDataSet)
    
    # Pbdep have to be given!
    #if not pbdep:
    #    raise ValueError(('You need to give at least one ParameterSet'
    #                      'representing dependables'))
        
    # pbdep need to be a list or a tuple. If not, make it one
    if not isinstance(pbdep, list) and not isinstance(pbdep, tuple):
        pbdep = [pbdep]
    
    # If 'pbdep' is not in the 'params' dictionary, prepare an empty one
    if not 'pbdep' in body.params:
        body.params['pbdep'] = OrderedDict()
    
    # If 'obs' is not in the 'params' dictionary, prepare an empty one
    if not 'obs' in body.params:
        body.params['obs'] = OrderedDict()
    
    # If 'syn' is not in the 'params' dictionary, prepare an empty one
    if not 'syn' in body.params:
        body.params['syn'] = OrderedDict()
    
    # For all parameterSets in pbdep, add them to body.params['pbdep']. This
    # dictionary is itself a dictionary with keys the different contexts, and
    # each entry in that context (ordered) dictionary, has as key the reference
    # and as value the parameterSet.
    parsed_refs = []
    for parset in pbdep:
        
        # Build the names for the pbdep context (e.g. lcdep), the data context
        # (e.g. lcobs) and the synthetic context (e.g. lcsyn).
        context = parset.context
        data_context = context[:-3] + 'obs'
        res_context = context[:-3] + 'syn'
        
        # Perform basic checks to make sure all dictionaries exist (perhaps this
        # is the first time to add things)
        if not context in body.params['pbdep']:
            body.params['pbdep'][context] = OrderedDict()
        if not data_context in body.params['obs']:
            body.params['obs'][data_context] = OrderedDict()
        if not res_context in body.params['syn']:
            body.params['syn'][res_context] = OrderedDict()
        ref = parset['ref']
        
        # If the ref already exist, generate a new one but warn the user
        if ref in body.params['pbdep'][context]:
            #logger.warning(("Adding pbdeps: ref '{}' already exists, "
            #                "generating a new one (via UUID4)").format(ref))
            logger.warning(("Adding pbdeps: ref '{}' already exists, "
                            "overwriting existing one.".format(ref)))
            
            #ref = str(uuid.uuid4())
            parset['ref'] = ref
            
            # replace values from original one to this one, at least if
            # they are a member of take_defaults
            if take_defaults is not None:
                for key in take_defaults:
                    if key in body.params['pbdep'][context][ref]:
                        parset[key] = body.params['pbdep'][context][ref][key]
        
        # Add the parameterSet to the relevant dictionary
        # This might be a little over the top, but I'll also check if the thing
        # that is added is really a ParameterSet
        if not isinstance(parset, parameters.ParameterSet):
            raise ValueError(("Trying to add pbdep with ref={} but it's "
                             "not a ParameterSet").format(ref))
        body.params['pbdep'][context][ref] = parset
        
        # Prepare results if they were not already added by the data parser
        if not ref in body.params['syn'][res_context]:
            if not res_context in body.params['syn']:
                raise ValueError("Trying to add {} to syn. Are you sure you passed pbdeps?".format(context))
            if context in result_sets:
                result_set = result_sets[context]
            else:
                result_set = datasets.DataSet
            body.params['syn'][res_context][ref] = \
                              result_set(context=res_context, ref=ref)
            logger.debug(('Prepared results ParameterSet for context '
                         '{} (ref={})'.format(res_context, ref)))
        
        # Remember this reference, 'cause we want to report on what's been added
        parsed_refs.append(ref)
    
    # That's it
    return parsed_refs
    
    
def _parse_obs(body, data):
    """
    Attach obs to a body.
    
    For each dataset, we also add a 'syn' thingy.
    
    For more explanations, this function is very similar to
    :py:func:`_parse_pbdeps`, except that we check for correspondence between
    pbdeps and obs. Implicitly we assume that :py:func:`_parse_pbdeps` is called
    first: if there exists only one pbdep of the given type, and it has a
    different reference than the observations added here, the reference will be
    changed too match the pbdep. This is done because if only adding one dataset
    per type, users often forget about the references.
    
    @param body: Body to add data to
    @type body: Body
    @param data: data to add to the Body
    @type data: list
    """
    result_sets = dict(lcobs=datasets.LCDataSet,
                       rvobs=datasets.RVDataSet,
                       spobs=datasets.SPDataSet,
                       plobs=datasets.PLDataSet,
                       ifobs=datasets.IFDataSet)
    
    # Data needs to be a list. If not, make it one
    if not isinstance(data, list):
        data = [data]
    
    # If 'obs' is not in the 'params' dictionary, make an empty one, and do the
    # same with 'syn'
    if not 'obs' in body.params:
        body.params['obs'] = OrderedDict()
    if not 'syn' in body.params:
        body.params['syn'] = OrderedDict()
    
    # Get the list of the current references in the data, to check if the data
    # we are adding has a corresponding pbdep
    pbdep_refs = body.get_refs(per_category=True, include=('pbdep', ))
    
    # For all parameterSets in data, add them to body.params['obs']. This
    # dictionary is itself a dictionary with keys the different contexts, and
    # each entry in that context (ordered) dictionary has as key the refs and as
    # value the parameterSet.
    parsed_refs = []
    for parset in data:
        
        # Build the string names of the obs and syn contexts
        context = parset.context.rstrip('obs')
        data_context = parset.context
        res_context = context + 'syn'
        
        # If this category (lc, rv...) does not exist yet, add it
        if not data_context in body.params['obs']:
            body.params['obs'][data_context] = OrderedDict()
        if not res_context in body.params['syn']:
            body.params['syn'][res_context] = OrderedDict()
            
        ref = parset['ref']
        
        # Check if the reference is present in the Body. There should be a
        # corresponding pbdep somewhere!
        if not context in pbdep_refs:
            raise ValueError(("You can't add observations of type {} (with "
                              "reference {}) without adding pbdeps. Add a "
                              "pbdep of type {} with that reference "
                              "first").format(context, ref, context))
        if not ref in pbdep_refs[context]:
            logger.warning(("Adding obs with ref='{}', but no corresponding "
                            "pbdeps found. Attempting fix.").format(ref))
            
            # If we have only one dataset and only one pbdep, we can assume
            # they belong together (right?)
            if len(pbdep_refs[context]) == 1 and len(data) == 1:
                ref = pbdep_refs[context][0]
                parset['ref'] = ref
                logger.info(("Fix succeeded: {}obs ref fixed to "
                             "'{}'").format(context, ref))
            
            else:
                logger.info(("Fix failed, there is no obvious match between "
                            "the pbdeps and obs"))
                # If there is only one pbdep reference, assume the user just
                # forgot about setting it, and correct it. Otherwise, raise a
                # ValueError
                raise ValueError(("Adding {context}obs with ref='{ref}', but "
                              "no corresponding {context}deps found (syn "
                              "cannot be computed). I found the following "
                              "{context}dep refs: {av}").format(context=context,
                               ref=ref, av=", ".join(pbdep_refs[context])))
        
        # Check if the ref happens to exist outside of this category. If it
        # does, something strange is happening.
        for category in pbdep_refs:
            if category == context:
                continue
            if ref in pbdep_refs[category]:
                raise ValueError(("You cannot add obs with ref {} of category "
                                  "{}, there already exists one in category "
                                  "{}!").format(ref, context, category))
        
        # If the ref already exist, generate a new one. This should never
        # happen, so send a critical message to the user
        if ref in body.params['obs'][data_context]:
            #logger.warning(('Data parsing: ref {} already exists!'
            #              'Generating new one...'.format(ref)))
            logger.warning(('Data parsing: ref {} already exists!'
                          'overwriting existing one.'.format(ref)))
            #ref = str(uuid.uuid4())
            parset['ref'] = ref
            
        # Prepare results if they were not already added by the data parser
        if not ref in body.params['syn'][res_context]:
            try:
                if data_context in result_sets:
                    result_set = result_sets[data_context]
                else:
                    result_set = datasets.DataSet
                body.params['syn'][res_context][ref] = \
                         result_set(context=res_context, ref=ref)
            except KeyError:
                raise KeyError(("Failed parsing obs {}: perhaps not "
                                "an obs?").format(ref))
            logger.debug(('Prepared results ParameterSet for context '
                          '{} (ref={})'.format(res_context, ref)))
        
        # This might be a little over the top, but I'll also check if the thing
        # that is added is really a ParameterSet
        if not isinstance(parset, datasets.DataSet):
            raise ValueError(("Trying to add obs with ref={} but it's "
                             "not a DataSet (it's a {})").format(ref, 
                                                     parset.__class__.__name__))
        body.params['obs'][data_context][ref] = parset
        
        
        # In case you suffer from amnesia: remember the ref
        parsed_refs.append(ref)
        
    
    # That's it
    return parsed_refs

#}

class CallInstruct:
    """
    Pass on calls to other objects.
    
    This is really cool!
    
    But coolness is not a quality criterion... maybe this causes us more
    trouble than it's worth...
    """
    def __init__(self, function_name, bodies):
        """
        Remember the bodies and the name of the function to call.
        
        @param function_name: name of the function to call
        @type function_name: Python definition
        @param bodies: list of Bodies
        @type bodies: list
        """
        self.function_name = function_name
        self.bodies = bodies
    
    def __call__(self, *args, **kwargs):
        """
        Now try to call the functions on the bodies.
        
        C{args} and C{kwargs} are anything C{self.function_name} accepts.
        """
        return [getattr(body, self.function_name)(*args, **kwargs) \
                    if hasattr(body, self.function_name) \
                    else None \
                    for body in self.bodies]    
                    
                    
class Body(object):
    """
    Base class representing a Body in the Universe.
    
    A Body represent the base properties an object can have. It basically has
    a minimal mesh (record array ``mesh``), a container to hold parameters
    (OrderedDict ``params``) and a time stamp (float ``time`` or None). It also
    provides basic manipulations and calculations of a mesh:
    
    **Input and output**
    
    .. autosummary::
    
        to_string
        list
        save
        copy
        plot2D
        plot3D
        get_label
        set_label
    
    **Accessing parameters and ParameterSets**
        
    .. autosummary::
    
        walk
        walk_type
        walk_all
        
        get_refs
        get_parset
        get_synthetic
        get_data
        get_model
        get_adjustable_parameters
        set_values_from_priors
        
        reset
        reset_and_clear
        clear_synthetic
        add_obs
        remove_obs
        

    **Statistics**
    
    .. autosummary::
    
        get_logp
        get_chi2
        
    
    **Computations**
    
    .. autosummary::
    
        rotate_and_translate
        rotate
        translate
        compute
        compute_scale_or_offset
        detect_eclipse_horizon
        compute_centers
        compute_sizes
        compute_normals
        area
        get_coords
        add_preprocess
        add_postprocess
        preprocess
        postprocess
        ifm
        pl
        sp

    
    The equality operator (``==`` and ``!=``) is implemented and will return
    True only if the left and right hand side are of the same class and have
    the exact same structure and variables.
    
    Iterating over a Body with the ``for element in body`` paradigm will iterate
    over all ParameterSets in the ``params`` attribute.
    
    Adding to Bodies will create a BodyBag.
    
    The point is to initialize it with a record array. Then, extra information
    can be added, such as the com (centre-of-mass), angles. Functions
    representing the surface and doing surface subdivisions should be added
    by subclassing.
    
    Basic fields are C{center, size, triangle, normal_} and C{mu}. Additional
    fields are C{visible, hidden} and C{partial}.
    
    The original values are stored in fields preceded with C{_o_}, except for
    the limb angles C{mu}.
    
    Limb angles are computed when L{rotate} is called, which has the
    line-of-sight as an input argument (aside from the Euler angles and rotation
    convention name).
    
    **Example usage:**
    
    >>> verts = [(0,0.0,0),(0,0.2,0),(1,0.3,0),(1,0.0,0),
    ...          (0,0.0,2),(0,0.3,2),(1,0.2,2),(1,0.0,2)]
    >>> tris = [np.hstack([verts[0],verts[1],verts[2]]),
    ...         np.hstack([verts[0],verts[2],verts[3]]),
    ...         np.hstack([verts[5],verts[2],verts[1]]),
    ...         np.hstack([verts[5],verts[6],verts[2]]),
    ...         np.hstack([verts[0],verts[4],verts[1]]),
    ...         np.hstack([verts[5],verts[1],verts[4]]),
    ...         np.hstack([verts[2],verts[7],verts[3]]),
    ...         np.hstack([verts[2],verts[6],verts[7]]),
    ...         np.hstack([verts[4],verts[7],verts[5]]),
    ...         np.hstack([verts[5],verts[7],verts[6]]),
    ...         np.hstack([verts[0],verts[3],verts[4]]),
    ...         np.hstack([verts[3],verts[7],verts[4]])]
    >>> tris = np.array(tris)
    >>> sizes = 0.5*np.ones(len(tris))
    >>> mesh = np.rec.fromarrays([np.array(tris),sizes],dtype=[('triangle','f8',(9,)),('size','f8',1)])
    
    
    We initiate the Body object, compute normals and centers, and recompute the
    sizes of the triangles.
    
    >>> body = Body(mesh,compute_centers=True,compute_normals=True,compute_sizes=True)
    
    We can easily plot the visible triangles:
    
    >>> m = mlab.figure()
    >>> body.plot3D(normals=True,scale_factor=0.5)
    
    ]include figure]]images/universe_body_0001.png]
    
    Incline 10 degrees wrt the line of sight and plot
    
    >>> body.rotate(incl=10./180.*pi)
    >>> m = mlab.figure()
    >>> body.plot3D(normals=True,scale_factor=0.5)
    
    ]include figure]]images/universe_body_0002.png]
    
    Rotate 10 degrees around the vertical axis and plot
    
    >>> body.rotate(theta=10./180.*pi)    
    >>> m = mlab.figure()
    >>> body.plot3D(normals=True,scale_factor=0.5)
    
    ]include figure]]images/universe_body_0003.png]
    
    Rotate 10 degrees in the plane of the sky and plot
    
    >>> body.rotate(Omega=10./180.*pi)
    >>> m = mlab.figure()
    >>> body.plot3D(normals=True,scale_factor=0.5)
    
    ]include figure]]images/universe_body_0004.png]
    """
    _params_tree = dict()
    
    def __init__(self, data=None, dim=3, orientation=None,
                 eclipse_detection='hierarchical',
                 compute_centers=False, compute_normals=False,
                 compute_sizes=False):
        """
        Initialize a Body.
        
        I'm too lazy to explain all the parameters here. Most of them can
        be left to the default value anyway. Have a look at examples in this
        very module for more info.
        
        @param eclipse_detection: takes a name of an algorithm
        (e.g. 'hierarchical' or 'simple')
        @type eclipse_detection: str
        """
        # We need to know the time and the dimensions (*sigh*... that'll be 3
        # always. I thought I was being general by foreseeing the possibility
        # to model 2D objects, but seriously... who does that?)
        self.time = None
        self.dim = dim
        
        # Remember how to detect eclipses for the Body. This will probably be
        # overriden in superclasses.
        self.eclipse_detection = eclipse_detection
        
        # Make a Body iterable, and give it a default unique label.
        self.index = 0
        self.label = str(id(self))
        
        # Bodies can be stacked hierarchically in BodyBags. Keep track of the
        # parent
        self.parent = None
        
        # Probably upon initialisation, the mesh is unknown. But we foresee the
        # possibility to initialize a Body with a custom mesh
        if data is None:
            n_mesh = 0
        else:
            n_mesh = len(data)
        
        # The basic float type is 'f8'
        ft = 'f8'
        
        # Initialise the mesh
        mesh = np.zeros(n_mesh, dtype=[('_o_center', ft, (dim, )),
                                  ('_o_size', ft),
                                  ('_o_triangle', ft, (3*dim, )),
                                  ('_o_normal_', ft,(dim, )),
                                  ('center', ft, (dim, )), ('size', ft),
                                  ('triangle', ft, (3*dim, )),
                                  ('normal_', ft, (dim, )),
                                  ('_o_velo___bol_', ft, (dim, )),
                                  ('velo___bol_', ft, (dim, )), ('mu', ft),
                                  ('partial', bool), ('hidden',bool),
                                  ('visible', bool)])
        
        # We allow the user to supply a custom mesh. In that case it needs to
        # be a record array
        if data is not None:
            # Only copy basic fields
            init_fields = set(mesh.dtype.names)
            fields_given = set(data.dtype.names)
            fields_given_extra = list(fields_given - init_fields)
            fields_given_basic = fields_given & init_fields
            
            # Append extra fields
            if fields_given_extra:
                mesh = pl.mlab.rec_append_fields(mesh,fields_given_extra,\
                                  [data[field] for field in fields_given_extra])
            
            # Take care of original values and their mutable counter parts.
            for field in fields_given_basic:
                ofield = '_o_{}'.format(field)
                
                if ofield in fields_given:
                    mesh[ofield] = data[ofield]
                elif ofield in mesh.dtype.names:
                    mesh[ofield] = data[field]
                
                mesh[field] = data[field]
        
        # In any case, we have a mesh now. Set it as an attribute
        self.mesh = mesh
        
        # If no information on visibility of the triangles is given, set them
        # all to be visible
        if data is None or not 'visible' in fields_given:
            self.mesh['visible'] = True
        
        # Compute extra information upon request
        if compute_centers:
            self.compute_centers()
        
        if compute_normals:
            self.compute_normals()
        
        if compute_sizes:
            self.compute_sizes()
        
        # Keep track of the current orientation, the original (unsubdivided)
        # mesh and all the parameters of the object.
        self.orientation = dict(theta=0, incl=0, Omega=0, pivot=(0, 0, 0),
                           los=[0, 0, +1], conv='YXZ', vector=[0, 0, 0])
        self.subdivision = dict(orig=None, mesh_args=None, N=None)
        self.params = OrderedDict()
        
        # The following attribute is deprecated, and was meant to store default
        # plot information. When we have unit tests in place, we can try to
        # remove this statement.
        self._plot = {'plot3D':dict(rv=(-150, 150))}
        
        # The following list of functions will be executed before and after a
        # call to set_time
        self._preprocessing = []
        self._postprocessing = []
        
        # We definitely need signals and a label, even if it's empty
        self.signals = {}
        self.label = None
        self.parent = None
        
        # Add a dict that we can use to store temporary information
        self._clear_when_reset = dict()
        self._main_period = dict()
        self._extra_mesh_columns = [] # e.g. ['B_', 'f8', (3,))] or [('abun','f8')]
        
    
    def __eq__(self, other):
        """
        Two Bodies are equal if all of their attributes are equal.
        """
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        """
        Two Bodies are different if not all of their attributes are equal.
        """
        return not self.__eq__(other)
    
    def __str__(self):
        """
        String representation of a Body.
        """
        return self.to_string()
    
    
    def fix_mesh(self):
        """
        Fix the mesh.
        
        This function is a dummy function; a single Body doesn't need to fix
        its mesh (there cannot be a clash in column names as for a BodyBag).
        However, the function is here for completeness. Should be overloaded.
        """
        return None
    
    
    def to_string(self,only_adjustable=False):
        """
        String representation of a Body.
        
        @param only_adjustable: only return the adjustable parameters
        @type only_adjustable: bool
        @return: string representation of the parameterSets
        @rtype: str
        """
        level1 = "============ {}/{}/{:d} ============\n{}\n"
        level2 = "============ {}/{}/{:d} ({}) ============\n{}\n"
        level3 = "============ {} ============\n{}\n"
        level4 = "============ {}/{:d} ============\n{}\n"
        
        txt = ''
        params = self.params
        for param in params:
            if isinstance(params[param], dict):
                for pbdep in params[param]:
                    
                    if isinstance(params[param][pbdep], list):
                        for i,ipbdep in enumerate(params[param][pbdep]):
                            txt += level1.format(param, pbdep, i,\
                              ipbdep.to_string(only_adjustable=only_adjustable))
                    
                    elif isinstance(params[param][pbdep], dict):
                        for i,lbl in enumerate(params[param][pbdep]):
                            txt += level2.format(param, pbdep, i, lbl,\
           params[param][pbdep][lbl].to_string(only_adjustable=only_adjustable))
                    
                    else:
                        txt += level3.format(param,\
                       params[param].to_string(only_adjustable=only_adjustable))

            elif isinstance(params[param],list):
                for i,ipbdep in enumerate(params[param]):
                    txt += level4.format(param, i,\
                              ipbdep.to_string(only_adjustable=only_adjustable))
            elif params[param] is not None:
                txt += level3.format(param,\
                       params[param].to_string(only_adjustable=only_adjustable))
        
        return txt
    
    def __iter__(self):
        """
        Make the class iterable.
        
        Iterating a body iterates through the parameters.
        """
        for param in list(self.params.values()):
            yield param
    
    
    def reset(self):
        """
        Reset the Body but do not clear the synthetic calculations.
        
        After a reset, calling C{set_time} again guarentees that the mesh
        will be recalculated. This could be useful if you want to change
        some basic parameters of an object and force the recomputation of
        the mesh, without the need for creating a new class instance.
        """
        self.time = None
        self._clear_when_reset = dict()
        self.subdivision['orig'] = None
        
        # Forget about the volume in binary stars
        if 'component' in self.params:
            _ = self.params['component'].pop_constraint('volume', None)
    
    
    def set_params(self, params, force=True):
        """
        Assign a parameterSet to a Body
        
        :param params: new ParameterSet to assign
        :type params: 
        """
        # We don't store subcontexts in the main root
        this_context = params.get_context().split(':')[0]
        
        # Some parameterSets are stored in lists, other ones are unique
        if this_context in self._params_tree and self._params_tree[this_context] == 'list':
            i_am_list = True
            params = [params]
        else:
            i_am_list = False
        
        # normal ParameterSet
        if this_context[-3:] not in ['obs', 'dep', 'syn']:
            if force or not (this_context in self.params):
                if i_am_list and this_context in self.params:
                    self.params[this_context] += params
                else:
                    self.params[this_context] = params
            elif (this_context in self.params) and i_am_list:
                self.params[this_context] += params
            else:
                raise ValueError('Cannot set ParameterSet to Body since it already exists (set force=True if you want to add it')
        
        # data-related stuff (not lists!)
        else:
            category = this_context[:-3]
            this_type = this_context[-3:]
            
            if this_type == 'dep':
                self.add_pbdeps(params)
            elif this_type == 'obs':
                self.add_obs(params)
            # special case for synthetics    
            else:    
                ref = params['ref']
            
                if not this_context in self.params[this_type]:
                    self.params[this_type][this_context] = OrderedDict()

                if force or not (this_context in self.params[this_type][this_context]):
                    self.params[this_type][this_context][ref] = params
                else:
                    raise ValueError('Cannot set PS to Body since it already exists (set force=True if you want to add it')
                return None
        
        
    def reset_and_clear(self):
        """
        Reset the Body and clear the synthetic calculations.
        """
        self.reset()
        self.clear_synthetic()
        
        
    def remove_ref(self, ref=None):
        """
        Remove all pbdep, syn and obs with a given reference.
        
        If no ref is given, everything is removed.
        """
        for dtype in ['syn', 'pbdep', 'obs']:
            if hasattr(self,'params') and dtype in self.params:
                for pbdeptype in self.params[dtype]:
                    for iref in self.params[dtype][pbdeptype]:
                        if ref is not None and iref!=ref:
                            continue
                        self.params[dtype][pbdeptype].pop(iref)
                    
    def change_ref(self, from_, to_):
        """
        Change ref from something to something else.
        """
        for dtype in ['syn', 'pbdep', 'obs']:
            if hasattr(self,'params') and dtype in self.params:
                for pbdeptype in self.params[dtype]:
                    for iref in self.params[dtype][pbdeptype]:
                        if from_ != iref:
                            continue
                        # this looses dictionary order. otherwise you have to
                        # build it anew (which is possibly but I have no time
                        # at the moment)
                        self.params[dtype][pbdeptype][to_] = self.params[dtype][pbdeptype].pop(from_)
                        self.params[dtype][pbdeptype][to_]['ref'] = to_
    
    def walk(self):
        """
        Walk through all the ParameterSets of a (nested) Body.
        
        This will recursively return all Parametersets in the Body.
        
        @return: generator of ParameterSets
        @rtype: generator
        """
        return utils.traverse(self,list_types=(BodyBag,Body,list,tuple),
                              dict_types=(dict,))
    
    
    def walk_type(self,type='syn'):
        """
        Walk through all types of a certain parameterSet.
        
        This can be handy to walk through all 'syn', 'pbdep' or 'obs'.
        
        @param type: type of dependable to walk through
        @type type: str, one of 'syn', 'pbdep', 'obs'
        @return: generator of ParameterSets
        @rtype: generator
        """
        if type in self.params:
            for param in list(self.params[type].values()):
                for value in param.values():
                    yield value
    
    def walk_dataset(self, type='syn'):
        for val,path in utils.traverse_memory(self,
                                     list_types=(Body, list,tuple),
                                     dict_types=(dict, )):
            if not isinstance(val, datasets.DataSet):
                continue
            # Only remember bodies
            path = [entry for entry in path if isinstance(entry, Body)]
                        
            # All is left is to return it
            yield path, val
    
    def walk_pbdep(self):
        for val,path in utils.traverse_memory(self,
                                     list_types=(Body, list,tuple),
                                     dict_types=(dict, )):
            if not isinstance(val, parameters.ParameterSet):
                continue
            elif val.get_context()[-3:]!='dep':
                continue
            # Only remember bodies
            path = [entry for entry in path if isinstance(entry, Body)]
                        
            # All is left is to return it
            yield path, val
    
    def walk_all(self,path_as_string=True):
        """
        Walk through all Bodies/ParameterSets/Parameters in nested Body(Bag).
        
        We need to:
        
            1. Walk through the BodyBags
            2. Walk through the Bodies
            3. Walk through the ParameterSets.
            
        And remember at what level we are!
        
        Each iteration, this function returns a path and a object (Body,
        ParameterSet, Parameter).
        
        We have a serious issue with this function: it should actually be
        defined outside of the Body. Right now, you **have** to iterate completely
        over it, i.e. not use the break statement. Otherwise, next time you
        iterate you start from there..
        """
        for val,path in utils.traverse_memory(self,
                                     list_types=(Body, list,tuple),
                                     dict_types=(dict, ),
                                       parset_types=(parameters.ParameterSet, ),
                                     get_label=(Body, ),
                                     get_context=(parameters.ParameterSet, ),
                                     skip=()):
            
            # First one is always root
            path[0] = str(self.__class__.__name__)
            
            # Convert to a string if desirable
            if path_as_string:
                for i in range(len(path)):
                    if isinstance(path[i], parameters.ParameterSet):
                        path[i] = path[i].context
                    elif isinstance(path[i], parameters.Parameter):
                        path[i] = path[i].get_qualifier()
                    elif isinstance(path[i], Body):
                        path[i] = path[i].get_label()
                    elif isinstance(path[i], str):
                        continue
                    else:
                        path[i] = '>>>'
            
            # All is left is to return it
            yield path, val
    
    def get_parent(self):
        """
        Return the parent of this Body.
        
        If it has a parent, return it. Otherwise, return None
        """
        return self.parent
    
    def set_parent(self, parent):
        """
        Set the parent of this Body.
        """
        self.parent = parent
        
    def get_sibling(self):
        """
        Return the other component in the same orbit
        
        If it does not have a parent, will return None
        """
        parent = self.get_parent()
        if parent is None: return None
        
        component = self.get_component()
        return [child for child in parent.get_children() if child.get_component()!=component][0]
    
    def get_globals(self, context='position'):
        """
        Return a global ParameterSet if possible, otherwise return None.
        
        We recursively walk up the BodyBag, until we encounter a ParameterSet
        of context ``globals``. If we find one, we return it and stop the
        iteration process.
        """
        # First check if there is global parameterSet here
        if context in self.params:
            return self.params[context]
        
        # Then walk up the parents.
        myparent = self.get_parent()
        if myparent is not None:
            return myparent.get_globals(context)
        
        # Else, for clarity, explicitly return None.
        else:
            return None
    
    def get_period(self):
        """
        Extract the main period and ephemeris from a Body.
        
        This can then be used to convert phases to time and vise versa.
        """
        period = self._main_period.get('period', parameters.Parameter(qualifier='period', value=np.inf, unit='d'))
        shift = self._main_period.get('shift', parameters.Parameter(qualifier='shift', value=0.0, unit='d'))
        t0 = self._main_period.get('t0', parameters.Parameter(qualifier='t0', value=0.0, unit='d'))
        return period.get_value('d'), t0.get_value('d'), shift.get_value('d')
    
    
    def set_period(self, period=None, t0=None, shift=None):
        """
        Set new Parameters for the system's period, t0 or shift.
        
        :param period: new period of the system
        :type period: Parameter
        :param t0: new t0 of the system
        :type t0: Parameter
        :param shift: new shift of the system
        :type shift: Parameter
        """
        if period is not None:
            self._main_period['period'] = period
        
        if t0 is not None:
            self._main_period['t0'] = t0
        
        if shift is not None:
            self._main_period['shift'] = shift
    
    def get_orbits(self, orbits=[], components=[]):
        """
        Return a list of all orbits this Body is in, and which components.
        
        A Body nested in a hierarchical system can be a member of many orbits,
        and for each orbit it could be the primary or secondary component.
        
        This function returns a list of the orbits, and a list of which
        components this body is in each of them. It is called hierarchically,
        return the inner most orbit first.
        """
        # Perhaps there is no orbit defined here: then do nothing
        if 'orbit' in self.params:
            orbit = self.params['orbit']
            # Perhaps orbit is None; then do nothing. Else, get the component,
            # and add the orbit and component to the function's attribute
            if orbit is not None:
                orbits.append(orbit)
                components.append(self.get_component())
                
        # After figuring out this orbit/component, go up to the parent and
        # do the same.
        myparent = self.get_parent()
        # Well, that is, if there is a parent!
        if myparent is not None:
            return myparent.get_orbits()
        
        # Else, return the orbit and component list, but make sure to clean
        # the function attributes for the next use of this function.
        else:
            # Return copies!
            retvalue = orbits[:], components[:]
            # And clear the function arguments
            while orbits:
                orbits.pop()
            while components:
                components.pop()
            return retvalue
            
    
    def compute(self, *args, **kwargs):
        """
        Compute synthetics to match the attached observations.
        
        See :py:func:`observatory.compute <phoebe.backend.observatory.compute>`
        for a thorough discussion of the arguments and keyword arguments.
        """
        observatory.compute(self,*args,**kwargs)
    
    
    def compute_pblum_or_l3(self):
        # Run over all pbdeps and see for which stars the pblum needs to be
        # computed
        reference_plum = {}
        
        for path, pbdep in self.walk_pbdep():
            
            if 'computed_pblum' in pbdep:
                pbdep.remove_constraint('computed_pblum')
            if 'computed_scaling' in pbdep:
                pbdep.remove_constraint('computed_scaling')
                        
            # Compute pblum here if needed
            if 'pblum' in pbdep and len(path):
                this_body = path[-1]
                this_ref = pbdep['ref']
            
                # The first passband luminosity needs to be the reference one,
                # if any others need to be scaled according to it
                if not this_ref in reference_plum and pbdep['pblum'] > -1:
                    passband_lum = this_body.luminosity(ref=this_ref) / constants.Rsol**2
                    reference_plum[this_ref] = pbdep['pblum'] / passband_lum
                # if the first one is -1, nothing needs to be computed at all
                elif not this_ref in reference_plum: # and thus pbep['pblum'] == 1
                    continue
                # otherwise we need to rescale the current one
                else:
                    passband_lum = this_body.luminosity(ref=this_ref) / constants.Rsol**2
                    
                # Now we need the synthetics so that we can scale them.
                syn = this_body.get_synthetic(ref=this_ref)
                
                # Each synthetic needs a difference scaling
                if syn.context in ['spsyn','plsyn'] and 'flux' in syn:
                    scale_column = 'flux'
                elif syn.context == 'lcsyn' and 'flux' in syn:
                    scale_column = 'flux'
                elif syn.context == 'rvsyn' and 'rv' in syn:
                    scale_column = 'rv'
                else:
                    logger.error('PBLUM/L3: skipping {}'.format(syn.context))
                    continue
                
                # Take the column to scale and scale it
                model = np.array(syn[scale_column])

                if pbdep['pblum'] >= 0:
                    model = model * pbdep['pblum'] / passband_lum
                    logger.info("Pblum of {} is forced to {}".format(this_ref, pbdep['pblum']))
                    pbdep.add_constraint("{{computed_pblum}} = {:.16e}".format(pbdep['pblum']))
                    pbdep.add_constraint("{{computed_scaling}} = {:.16e}".format(pbdep['pblum'] / passband_lum))
                else:
                    
                    model = model * reference_plum[this_ref]
                    logger.info("Pblum of {} is computed to {}".format(this_ref, reference_plum[this_ref]*passband_lum))
                    pbdep.add_constraint("{{computed_pblum}} = {:.16e}".format(reference_plum[this_ref]*passband_lum))
                    pbdep.add_constraint("{{computed_scaling}} = {:.16e}".format(reference_plum[this_ref]))

                
    
    def set_pblum_or_l3(self):
        do_continue = False
        for path, pbdep in self.walk_pbdep():
            
            if not path:
                continue
            
            this_body = path[-1]
            this_ref = pbdep['ref']
            
            has_computed_pblum = 'computed_pblum' in pbdep.constraints
            has_nonzero_l3 = 'l3' in pbdep and pbdep['l3'] > 0
            
            
            # Correct for pblum/l3 here if needed
            if not do_continue and (has_computed_pblum or has_nonzero_l3):
                
                if has_computed_pblum:
                    scale = pbdep.request_value('computed_scaling')
                else:
                    scale = None
                 
                # Now we need the synthetics so that we can scale them.
                syn = this_body.get_synthetic(ref=this_ref)
                
                # Each synthetic needs a difference scaling
                if syn.context in ['spsyn','plsyn'] and 'flux' in syn:
                    scale_column = 'flux'
                elif syn.context == 'lcsyn' and 'flux' in syn:
                    scale_column = 'flux'
                elif syn.context == 'rvsyn' and 'rv' in syn:
                    scale_column = 'rv'
                else:
                    logger.error('PBLUM/L3: skipping {}'.format(syn.context))
                    continue
                
                # and remember the value
                the_scale_column = np.array(syn[scale_column])
                
                if scale is not None:
                    the_scale_column = scale * the_scale_column
            
                # correct for l3. l3 is in units of whatever pblum is (i.e. if
                # there is no rescaling, it is in absolute fluxes, otherwise it
                # is in units of *this* pblum)
                if has_nonzero_l3:
                    the_scale_column = pbdep['l3'] + the_scale_column
                
                # Convert back to a list
                syn[scale_column] = list(the_scale_column)            
            
    
    
                
    def compute_scale_or_offset(self):
        """
        Compute and set passband luminosity and third light if required.
        
        We need to be able to compute pblum and l3 for all data individually
        if we wish, but we might also want to force certain data to have the
        same pblum and/or l3 (i.e link them). This could make sense for SED
        fits, where pblum is then interpreted as some kind of scaling factor
        (e.g. the angular diameter). Perhaps there are other applications.
        
        The pblum and l3 (scale factor and offset value) have to be stored in
        the observation set, because the synthetic sets can be virtual: for
        example for light curves, there is no "one" synthetic light curve,
        every component has its own from which the total one is built.
        
        See :py:func:`compute_scale_or_offset` for more information.
        """
        link = None
        # We need observations of course
        if not 'obs' in self.params:
            logger.info('Cannot compute scale or offset, no observations defined')
            return None
        
        # We'll collect the complete model first (i.e. all the observations in
        # one big 1D array). We'll keep track of the references, so that we know
        # which points represent a particular observation set. Then afterwards,
        # we compute the pblums for all the linked datasets... or for all of
        # separately if link=None.
        
        # Possibly some data need to be grouped
        groups = dict()
        
        for path, obs in self.walk_dataset():
            if not obs.get_context()[-3:] == 'obs':
                continue
            
            if len(path):
                subsystem = path[-1]
            else:
                subsystem = self
            
            # Ignore disabled datasets
            if not obs.get_enabled():
                continue
            
            # Get the model corresponding to this observation
            syn = subsystem.get_synthetic(category=obs.context[:-3],
                                       ref=obs['ref'],
                                       cumulative=True)

            # Make sure to have loaded the observations from a file
            loaded = obs.load(force=False)
            
            # Get the "model" and "observations" and their error.
            if obs.context in ['spobs','plobs'] and 'flux' in obs:
                model = np.array(syn['flux'])/np.array(syn['continuum'])
                obser = np.array(obs['flux']) / np.array(obs['continuum'])
                sigma = np.array(obs['sigma'])
            
            elif obs.context == 'lcobs' and 'flux' in obs:
                model = np.array(syn['flux'])
                obser = np.array(obs['flux'])
                sigma = np.array(obs['sigma'])
            
            elif obs.context == 'rvobs' and 'rv' in obs and 'sigma' in obs:
                model = np.array(syn['rv'])
                obser = np.array(obs['rv']/constants.Rsol*(24*3.6e6))
                sigma = np.array(obs['sigma']/constants.Rsol*(24*3.6e6))
            
            elif obs.context == 'ifobs' and 'vis2' in obs:
                # this works if no closure phases are available
                model = np.array(syn['vis2'])
                obser = np.array(obs['vis2'])
                sigma = np.array(obs['sigma_vis2'])
                
            else:
                logger.error('SCALE/OFFSET: skipping {}'.format(obs.context))
                continue
            
            # It is possible that the pblum and l3 are linked to other
            # datasets, e.g. to determine a scaling factor of all
            # multicolour photometry (but not together with a possible
            # light curve).
            if 'group' in obs:
                this_group = obs['group']
                
                # Keep track of model, observations, sigma and obs dataset
                # itself (to fill in pblum and l3 in all of them)
                if not this_group in groups:
                    groups[this_group] = [[],[],[],[]]
                groups[this_group][0].append(model)
                groups[this_group][1].append(obser)
                groups[this_group][2].append(sigma)
                groups[this_group][3].append(obs)
                continue
                
            
            # Determine pblum and l3 for these data if necessary. The pblum
            # and l3 for the model, independently of the observations,
            # should have been computed before when computing the model.
            # Only fit the pblum and l3 here if these parameters are
            # available in the dataset, and they are set to be adjustable
            do_scale = False
            do_offset = False
            preset_scale = 1.0 if not 'scale' in obs else obs['scale']
            preset_offset = 0.0 if not 'offset' in obs else obs['offset']
            
            if 'scale' in obs and obs.get_adjust('scale') and not obs.has_prior('scale'):
                do_scale = True
                preset_scale = 1.0
            
            if 'offset' in obs and obs.get_adjust('offset') and not obs.has_prior('offset'):
                do_offset = True
                preset_offset = 0.0
            
            # Do the computations
            if do_scale or do_offset:
                
                # We allow for negative coefficients in spectra
                if obs.context in ['plobs','spobs', 'rvobs']:
                    alg = 'lstsq'
                
                # But not in other stuff
                else:
                    alg = 'nnls'
                    
                scale, offset = compute_scale_or_offset(model,
                               (obser - preset_offset)/preset_scale,
                               (sigma/preset_scale), scale=do_scale, offset=do_offset, type=alg)
            
            #   perhaps we don't need to fit, but we still need to
            #   take it into account
            if not do_scale and 'scale' in obs:
                scale = obs['scale']
            elif not do_scale:
                scale = 1.0
            if not do_offset and 'offset' in obs:
                offset = obs['offset']
            elif not do_offset:
                offset = 0.0
            #-- set the values and add them to the posteriors
            if do_scale:
                obs['scale'] = scale
            if do_offset:
                obs['offset'] = offset
            if loaded:
                obs.unload()
            
            msg = '{}: scale={:.6g} ({}), offset={:.6g} ({})'
            logger.info(msg.format(obs['ref'], scale,\
                        do_scale and 'computed' or 'fixed', offset, do_offset \
                        and 'computed' or 'fixed'))
        
        # Now we can compute the scale and offset's for all groups
        if groups:
            for group in groups:
                
                # Merge all data
                model = np.hstack(groups[group][0])
                obser = np.hstack(groups[group][1])
                sigma = np.hstack(groups[group][2])
                obs = groups[group][3][0]
                
                # Only first obs is checked, but they should all be the same
                do_scale = False
                do_offset = False
                
                if 'scale' in obs and obs.get_adjust('scale'):
                    do_scale = True
                
                if 'offset' in obs and obs.get_adjust('offset'):
                    do_offset = True
                
                # Do the computations
                if do_scale or do_offset:
                    
                    # We allow for negative coefficients in spectra
                    if obs.context in ['plobs','spobs']:
                        alg = 'lstsq'
                    
                    # But not in other stuff
                    else:
                        alg = 'nnls'
                    scale, offset = compute_scale_or_offset(model, obser, sigma, 
                                   scale=do_scale, offset=do_offset, type=alg)
                
                # perhaps we don't need to fit, but we still need to take it
                # into account
                if not do_scale and 'scale' in obs:
                    scale = obs['scale']
                elif not do_scale:
                    scale = 1.0
                if not do_offset and 'offset' in obs:
                    offset = obs['offset']
                elif not do_offset:
                    offset = 0.0
                # Set the values for all observations
                for obs in groups[group][3]:
                    if do_scale:
                        obs['scale'] = scale
                    if do_offset:
                        obs['offset'] = offset
                msg = 'Group {} ({:d} members): scale={:.6g} ({}), offset={:.6g} ({})'
                logger.info(msg.format(group, len(groups[group][3]),scale,\
                            do_scale and 'computed' or 'fixed', offset, do_offset \
                            and 'computed' or 'fixed'))
    
    def luminosity(self, ref='__bol', numerical=False):
        """
        Compute the luminosity of this body.
        """
        return luminosity(self, ref=ref, numerical=numerical)
    
    def bin_oversampling(self):
        # Iterate over all datasets we have
        for path, syn in self.walk_dataset():
            if not syn.get_context()[-3:] == 'syn':
                continue
            if len(path):
                subsystem = path[-1]
            else:
                subsystem = self
            
            # Make sure to have loaded the observations from a file
            loaded = syn.load(force=False)
            
            if hasattr(syn, 'bin_oversampling'):
                syn.bin_oversampling()
                
            if loaded:
                syn.unload()
    
    def get_logp(self, include_priors=False):
        r"""
        Retrieve probability or goodness-of-fit.
        
        If the datasets have passband luminosities C{pblum} and/or third
        light contributions ``l3``, they will be:
        
            - included in the normal fitting process if they are adjustable
              and have a prior
            - linearly fitted after each model computation, but otherwise not
              included in the fitting process if they are adjustable but do not
              have a prior
            - left to their original values if they are not adjustable.
        
        Every data set has a statistical weight, which is used to weigh them
        in the computation of the total probability.
        
        If ``statweight==0``, then:
        
        .. math::
            :label: prob
        
            p = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2_{y_i}}}
            \exp\left(-\frac{(y_i - 
            \mathrm{pblum}\cdot m_i - l_3)^2}{2\sigma_{y_i}^2}\right)
            
            \log p = \sum_{i=1}^N -\frac{1}{2}\log(2\pi) - \log(\sigma_{y_i})
            - \frac{(y_i - \mathrm{pblum}\cdot m_i - l_3)^2}{2\sigma_{y_i}^2}
        
        With :math:`y_i, \sigma_{y_i}` the data and their errors, and :math:`m_i`
        the model points. Here :math:`p` gives the expected frequency of getting
        a value in an infinitesimal range around :math:`y_i` per unit :math:`dy`.
        To retrieve the :math:`\chi^2`, one can observe that the above is
        equivalent to
        
        .. math::
        
            \log p = K -\frac{1}{2}\chi^2
            
        with :math:`K` an 'absorbing term'. Equivalently,
        
        .. math::
        
            K = \log p + \frac{1}{2}\chi^2
        
        or
        
        .. math::
            :label: chi2
        
            \chi^2 = 2 (K - \log p )
        
        Note that Eq. :eq:`chi2` is quite intuitive: since
        
            1. :math:`p` is expected to have a uniform distribution
            2. the negative natural log of a uniform distribution is expected to
               follow an exponential distribution
            3. an exponential distribution scaled with a factor of two results
               in a :math:`\chi^2` distributed parameter.
        
        If ``statweight=0`` (thus following the above procedure), combining
        different datasets should be naturally equivalent with `Fisher's method <http://en.wikipedia.org/wiki/Fisher's_method>`_
        of combining probabilities from different (assumed independent!) tests:
        
        .. math::
        
            \chi^2 = -2 \sum_i^k\log(p_i)
            
        With :math:`i` running over all different tests (i.e. datasets). However,
        the :math:`p` from Eq. :eq:`prob` is not a real probability, merely an
        expected frequency (or likelihood if you wish): it is a value from the
        probability *density* function. Imagine a measurement of an absolute
        flux in erg/s/cm2/:math:`\AA`, of the order of 1e-15 with an error of
        1e-17. Then the value of :math:`p` will be extremely large because of
        the appearance of the inverse :math:`\sigma` in the denominator. In
        principle this is not a problem since they all get observed in the
        factor :math:`K`, but it effectively makes the parameter space stretched
        or squeezed in some directions.
        
        If ``include_priors=True``, then also the distribution of the priors
        will be taken into account in the computation of the probability in 
        Eq. :eq:`prob`:
        
        .. math::
        
            p = \prod_{i=1}^{N_\mathrm{data}} \frac{1}{\sqrt{2\pi\sigma^2_{y_i}}}
                \exp\left(-\frac{(y_i - 
                \mathrm{pblum}\cdot m_i - l_3)^2}{2\sigma_{y_i}^2}\right)
                \prod_{j=1}^{N_\mathrm{pars}} P_j(s_j)
        
        Where :math:`P_j(s_j)` is the probability of the current value
        :math:`s_j` of the :math:`j` th parameter  according to its prior
        distribution. Note that even the non-fitted parameters that have a
        prior will be taken into account. If they are not fitted, they will not
        have an influence on the probability but for a constant term. On the
        other hand, one can exploit this property to define a parameter that
        *derives* it's value from other parameters (e.g. distance from ``pblum``
        or ``vsini`` from ``rotperiod`` and ``radius``),
        and take also that into account during fitting. One could argue about
        the statistical validity of such a procedure, but it might come in handy
        to simplify the fitting problem.
        
        If ``statweight>1`` each :math:`\log p` value will actually be the mean
        of all :math:`\log p` within one dataset, weighted with the value of
        ``statweight``. This an ugly hack to make some datasets more or less
        important, but is generally not a good approach because it involves
        a subjective determination of the ``statweight`` parameter.
        
        .. warning::
        
            The :math:`\log p` returned by this function is an
            **expected frequency** and not a true probability (it's value is not
            between 0 and 1). That is, the :math:`p` comes from the probability
            density function. To get the probability itself,
            you can use scipy on the :math:`\chi^2`:
                
            >>> n_data = 100
            >>> n_pars = 7
            >>> chi2 = 110.0
            >>> k = n_data - n_pars
            >>> prob = scipy.stats.distributions.chi2.cdf(chi2, k)
            >>> print(prob)
            0.889890617416
            
            If ``prob`` is close to 1 then your model is implausible, if it is
            close to zero it is very plausible.
        
        .. note:: See also
            
            :py:func:`get_chi2 <Body.get_chi2>` to compute the
            :math:`\chi^2` statistic and probability
        
        References: [Hogg2009]_.
        
        @return: log probability, chi square, Ndata
        @rtype: float, float, float
        """
        log_f = 0. # expected frequency
        log_p = 0. # probability 
        chi2 = []  # chi squares
        n_data = 0.
        
        # Run the check just to make sure all constraints on the parameters
        # are set correctly
        self.check()
        
        # Iterate over all datasets we have
        for path, obs in self.walk_dataset():
            if not obs.get_context()[-3:] == 'obs':
                continue
            
            if len(path):
                subsystem = path[-1]
            else:
                subsystem = self
            
            # Ignore disabled datasets
            if not obs.get_enabled():
                continue
            
            # Get the model corresponding to this observation
            modelset = subsystem.get_synthetic(category=obs.context[:-3],
                                          ref=obs['ref'],
                                          cumulative=True)
            
            # Perhaps calculations aren't done yet
            if not len(modelset):
                continue
            
            # Make sure to have loaded the observations from a file
            loaded = obs.load(force=False)
            
            # Get the "model" and "observations" and their error.
            if obs.context in ['spobs','plobs']:
                model = np.array(modelset['flux']) / np.array(modelset['continuum'])
                obser = np.array(obs['flux']) / np.array(obs['continuum'])
                sigma = np.array(obs['sigma'])
            
            elif obs.context == 'lcobs':
                model = np.array(modelset['flux'])
                obser = np.array(obs['flux'])
                sigma = np.array(obs['sigma'])
            
            elif obs.context == 'ifobs':
                model = np.array(modelset['vis2'])
                obser = np.array(obs['vis2'])
                sigma = np.array(obs['sigma_vis2'])
            
            elif obs.context == 'rvobs':
                #model = conversions.convert('Rsol/d', 'km/s', np.array(modelset['rv']))
                model = np.array(modelset['rv'])
                obser = np.array(obs['rv'])
                sigma = np.array(obs['sigma'])
                
            else:
                raise NotImplementedError(("probability for "
                             "{}").format(obs.context))
                
            # Take scale and offset into account:
            scale = obs['scale'] if ('scale' in obs) else 1.0
            offset = obs['offset'] if ('offset' in obs) else 0.0
            
            # Compute the log probability ---> not sure that I need to do
            #                                  sigma*scale, I'm not touching
            #                                  the observations!
            term1 = - 0.5*np.log(2*pi*(sigma)**2)
            term2 = - (obser-model*scale-offset)**2 / (2.*(sigma)**2)                       
            
            # Do also the Stokes V profiles. Because they contain the
            # derivative of the intensity profile, the offset factor disappears
            if obs.context == 'plobs':
                if 'V' in obs['columns']:
                    model = np.array(modelset['V']) / np.array(modelset['continuum'])
                    obser = np.array(obs['V']) / np.array(obs['continuum'])
                    sigma = np.array(obs['sigma_V'])
                    term1 += - 0.5*np.log(2*pi*(sigma)**2)
                    term2 += - (obser-model*scale)**2 / (2.*(sigma)**2)

            # Statistical weight:
            statweight = obs['statweight']
            
            #   if stat_weight is negative, we try to determine the
            #   effective number of points:
            # ... not implemented yet ...
            #   else, we take take the mean and multiply it with the
            #   weight:
            if statweight>0:
                this_logf = (term1 + term2).mean() * statweight
                this_chi2 = -(2*term2).mean() * statweight
            
            #   if statistical weight is zero, we don't do anything:
            else:
                this_logf = (term1 + term2).sum()
                this_chi2 = -2*term2.sum()
            
            logger.debug("scale = {:.3g}, offset = {:.3g}".format(scale, offset))
            #logger.info("Chi2 of {} = {}".format(obs['ref'], -term2.sum()*2))
            logger.info("Chi2 of {} = {} (statweight {})".format(obs['ref'], this_chi2, statweight))
            log_f += this_logf
            chi2.append(this_chi2)
            n_data += len(obser)
            if loaded:
                obs.unload()
            
        # Include priors if requested
        if include_priors:
            # Get a list of all parameters with priors
            pars = self.get_parameters_with_priors()
            
            for par in pars:
                # Get the log probability of the current value given the prior.
                prior = par.get_prior()
                value = par.get_value()
                pdf = prior.pdf(domain=value)[1]
                this_logf = np.log(pdf)
                log_f += this_logf
                if prior.distribution == 'normal':
                    mu = prior.distr_pars['mu']
                    sigma = prior.distr_pars['sigma']
                    this_chi2 = (value - mu)**2 / sigma**2
                else:
                    if pdf==0:
                        this_chi2 = np.inf#1e6#10*n_data
                    else:
                        this_chi2 = 0.0
                    
                chi2.append(this_chi2)
        
        # log_f is nan for whatever reason, it's actually -inf
        # then the chi2 should probably also be large...?
        if np.isnan(log_f):
            log_f = -np.inf
        
        return log_f, chi2, n_data
    
    
    def get_chi2(self, include_priors=False):
        r"""
        Return the :math:`\chi^2` and resulting probability of the model.
        
        If ``prob`` is close to unity, the model is implausible, if it is
        close to zero, it is very plausible.
        """
        # Get the necessary info
        logp, chi2, n_data = self.get_logp(include_priors=include_priors)
        adj = self.get_adjustable_parameters()
        n_par = len(adj)
        
        # Compute the chi2 probability with n_data - n_par degrees of freedom
        k = n_data - n_par
        prob = 1-scipy.stats.distributions.chi2.cdf(chi2, k)
        logprob = np.log(prob)
        logprob[np.isinf(logprob)] = -1e10
        total_chi2 = -2* np.sum(logprob)
        total_prob = scipy.stats.distributions.chi2.cdf(total_chi2, 2*len(prob))
        # That's it!
        return total_chi2, total_prob, n_data, n_par
        
    def check(self, return_errors=False):
        """
        Check if a system is OK.
        
        What 'OK' is, depends on a lot of stuff. Typically this function can be
        used to do some sanity checks when fitting, such that impossible systems
        can be avoided.
        
        We check if a parameter (or all) has a finite log likelihood.
        
        If ``qualifier=None``, all parameters with priors are checked. If any is
        found to be outside of bounds, ``False`` is returned. Any other parameter,
        even the ones without priors, are checked for their limits. If any is
        outside of the limits, ``False`` is returned. If no parameters are
        outside of their priors and/or limits, ``True`` is returned.
        
        We preprocess the system first.
        
        """
        
        self.preprocess()
        
        error_messages = []
        
        already_checked = []
        system = self.get_system()
        were_still_OK = True
        
        for parset in self.walk():
            
            # Run constraints
            parset.run_constraints()
            
            for par in parset:

                if not were_still_OK and not return_errors:
                    continue
                
                val = parset.get_parameter(par)
                        
                # If we've already checked this parameter, don't bother
                if val.get_unique_label() in already_checked:
                    continue
                
                # If the value is outside of the limits (if it has any), we are
                # not OK!
                if not val.is_inside_limits():
                    were_still_OK = False
                    error_messages.append('{}={} is outside of reasonable limits {}'.format(val.get_qualifier(),
                                                                                            val.get_value(),
                                                                                            val.get_limits()))
                    continue
                
                # If the value has zero probability, we're not OK!
                if val.has_prior() and np.isinf(val.get_logp()):
                    were_still_OK = False
                    error_messages.append('{}={} is outside of prior {}'.format(val.get_qualifier(),
                                                                         val.get_value(),
                                                                         val.get_prior()))
                    continue
            
                
                # Remember we checked this one
                already_checked.append(val.get_unique_label())
        
        if return_errors:
            return were_still_OK, error_messages
        else:
            return were_still_OK
            
            
    def get_data(self):
        """
        Return all data in one long chain of data.
        """
        mu = []
        sigma = []
        for idata in self.params['obs'].values():
            for observations in idata.values():
                
                # Ignore disabled datasets
                if not observations.get_enabled():
                    continue
                
                # Make sure to have loaded the observations from a file
                loaded = observations.load(force=False)
                if observations.context == 'spobs':
                    obser_ = np.ravel(observations['flux']/observations['continuum'])
                    sigma_ = np.ravel(observations['sigma'])
                elif observations.context == 'lcobs':
                    obser_ = np.ravel(observations['flux'])
                    sigma_ = np.ravel(observations['sigma'])
                elif observations.context == 'ifobs':
                    obser_ = np.ravel(observations['vis2'])
                    sigma_ = np.ravel(observations['sigma_vis2'])
                elif observations.context == 'rvobs':
                    obser_ = np.ravel(observations['rv'])
                    sigma_ = np.ravel(observations['sigma'])
                else:
                    raise NotImplementedError('probability')  
                
                # Append to the "whole" model.
                mu.append(obser_)
                sigma.append(sigma_)
                if loaded:
                    observations.unload()
        
        return np.hstack(mu), np.hstack(sigma)
    
    
    def get_model(self):
        """
        Return all data and complete model in one long chain of data.
        
        For some of the fitters, the output of this function is used in the
        fitting process. So if anything goes wrong, this is probably the place
        to start debugging.
        
        @return: obs, sigma, model
        @rtype: 3xarray
        """
        model = []
        mu = []
        sigma = []
        for idata in self.params['obs'].values():
            for observations in idata.values():
                
                # Ignore disabled datasets
                if not observations.get_enabled():
                    continue
                
                modelset = self.get_synthetic(category=observations.context[:-3],
                                         ref=observations['ref'],
                                         cumulative=True)
                
                # Make sure to have loaded the observations from a file
                loaded = observations.load(force=False)
                if observations.context in ['spobs', 'plobs']:
                    model_ = np.ravel(np.array(modelset['flux'])/np.array(modelset['continuum']))
                    obser_ = np.ravel(np.array(observations['flux'])/np.array(observations['continuum']))
                    sigma_ = np.ravel(np.array(observations['sigma']))
                elif observations.context == 'lcobs':
                    model_ = np.ravel(np.array(modelset['flux']))
                    obser_ = np.ravel(np.array(observations['flux']))
                    sigma_ = np.ravel(np.array(observations['sigma']))
                elif observations.context == 'ifobs':
                    model_ = np.ravel(np.array(modelset['vis2']))
                    obser_ = np.ravel(np.array(observations['vis2']))
                    sigma_ = np.ravel(np.array(observations['sigma_vis2']))
                elif observations.context == 'rvobs':
                    #model_ = conversions.convert('Rsol/d', 'km/s', np.ravel(np.array(modelset['rv'])))
                    model_ = np.ravel(np.array(modelset['rv']))
                    obser_ = np.ravel(np.array(observations['rv']))
                    sigma_ = np.ravel(np.array(observations['sigma']))
                else:
                    raise NotImplementedError('probability')
                
                # Statistical weight:
                statweight = observations['statweight']
                
                # Take scale and offset into account:
                scale = observations['scale'] if ('scale' in observations) else 1.0
                offset = observations['offset'] if ('offset' in observations) else 0.0
                model_ = scale*model_ + offset
                
                if observations.context == 'plobs':
                    if 'V' in observations['columns']:
                        # We need to correct the Stokes profile for the passband
                        # luminosity factor, as this was not taken into account
                        # during the calculations
                        model_ = np.hstack([model_, np.ravel(np.array(modelset['V'])/np.array(modelset['continuum'])*scale)])
                        obser_ = np.hstack([obser_, np.ravel(np.array(observations['V'])/np.array(observations['continuum']))])
                        sigma_ = np.hstack([sigma_, np.ravel(np.array(observations['sigma_V']))])
                
                # Transform to log if necessary:
                if 'fittransfo' in observations and observations['fittransfo']=='log10':
                    sigma_ = sigma_ / (obser_*np.log(10))
                    model_ = np.log10(model_)
                    obser_ = np.log10(obser_)
                    logger.info("Transformed model to log10 for fitting")
                
                # Append to the "whole" model.
                model.append(model_)
                mu.append(obser_)
                sigma.append(sigma_ / statweight**2)
                if loaded:
                    observations.unload()
                    
        if not len(mu):
            mu = np.array([])
            sigma = np.array([])
            model = np.array([])
            return mu, sigma, model
        else:
            return np.hstack(mu), np.hstack(sigma), np.hstack(model)
    
    
    def get_adjustable_parameters(self, with_priors=True):
        """
        Return a list of all adjustable parameters.
        
        :param with_priors: flag to take only adjustable parameters with priors (:envvar:`with_priors=True`),
         only adjustable parameters without priors (:envvar:`with_priors=False`) or all
         adjustable parameters(:envvar:`with_priors=None`).
        """
        ids = []
        mylist = []
        for path, val in self.walk_all():
            path = list(path)
            
            
            if isinstance(val,parameters.Parameter) and val.get_adjust() and not val.get_unique_label() in ids:
                # If include priors but this parameters has none, continue
                if with_priors is True and not val.has_prior():
                    continue
                # If not include priors but this parameters has one, continue
                elif with_priors is False and val.has_prior():
                    continue
                # If include priors is not set, add it anyway
                else:
                    mylist.append(val)
                    ids.append(val.get_unique_label())
                
                    
        return mylist
    
    
    def get_parameters_with_priors(self, is_adjust=None, is_derived=None):
        """
        Return a list of all parameters with priors.
        """
        mylist = []
        for path, val in self.walk_all():
            if isinstance(val,parameters.Parameter) and val.has_prior() and not val in mylist:
                if is_adjust is True and not val.get_adjust():
                    continue
                elif is_adjust is False and val.get_adjust():
                    continue
                elif is_derived is True and not val.get_replaced_by():
                    continue
                elif is_derived is False and val.get_replaced_by():
                    continue
                else:
                    mylist.append(val)
        return mylist
    
    def get_label(self):
        """
        Return the label of the class instance.
        
        @return: the label of the class instance
        @rtype: str
        """
        return self.label
    
    
    def set_label(self,label):
        """
        Set the label of the class instance.
        
        @param label: label of the class instance
        @type label: str
        """
        self.label = label
    
    
    def add_preprocess(self, func, *args, **kwargs):
        """
        Add a preprocess to the Body.
        
        The list of preprocessing functions are executed before set_time is
        called.
        
        @param func: name of a processing function in backend.processes
        @type func: str
        """
        # first check if the preprocess is already available
        available = [prep[0] for prep in self._preprocessing]
        
        # If it's available, overwrite
        if func in available:
            index = available.index(func)
            self._preprocessing[index] = (func, args, kwargs)
        
        # Else append
        else:
            self._preprocessing.append((func, args, kwargs))
    
    def remove_preprocess(self, func):
        """
        Remove preprocess from a Body.
        """
        # first check if the preprocess is available
        available = [prep[0] for prep in self._preprocessing]
        
        # If it's available, remove
        if func in available:
            index = available.index(func)
            thrash = self._preprocessing.pop(index)
        

    def add_postprocess(self, func, *args, **kwargs):
        """
        Add a postprocess to the Body.
        
        @param func: name of a processing function in backend.processes
        @type func: str
        """
        self._postprocessing.append((func, args, kwargs))
    
        
    def preprocess(self, time=None, **kwargs):
        """
        Run the preprocessors.
        
        @param time: time to which the Body will be set
        @type time: float or None
        """
        for func, arg, kwargs in self._preprocessing:
            getattr(processing, func)(self, time, *arg, **kwargs)
        
    
    
    def postprocess(self, time=None):
        """
        Run the postprocessors.
        """
        for func, args, kwargs in self._postprocessing:
            getattr(processing, func)(self, time, *args, **kwargs)
    
    
    def set_values_from_priors(self):
        """
        Set values from adjustable parameters with a prior to a random value
        from it's prior.
        """
        walk = utils.traverse(self,list_types=(BodyBag, Body, list, tuple),
                                   dict_types=(dict, ))
        for parset in walk:
            
            # For each parameterSet, walk through all the parameters
            for qual in parset:
                
                # Extract those which need to be fitted
                if parset.get_adjust(qual) and parset.has_prior(qual):
                    parset.get_parameter(qual).set_value_from_prior()
    
    
    
    
    #{ Functions to manipulate the mesh    
    def detect_eclipse_horizon(self, eclipse_detection=None, **kwargs):
        r"""
        Detect the triangles at the horizon and the eclipsed triangles.
        
        Possible C{eclipse_detection} algorithms are:
        
            1. **hierarchical**: full generic detection
            
            2. **simple**: horizon detection based on the direction of the
               :math:`\mu`-angle
        
        Extra kwargs are passed on to the eclipse detection algorithm.
        
        A possible kwarg is for example C{threshold} when
        C{eclipse_detection=False}.
        
        @param eclipse_detection: name of the algorithm to use to detect
         the horizon or eclipses
        @type eclipse_detection: str
        """
        if eclipse_detection is None:
            eclipse_detection = self.eclipse_detection
        
        # Generic eclipse detection
        if eclipse_detection == 'hierarchical':
            eclipse.detect_eclipse_horizon(self)
        
        # Simple eclipse detection -> maybe replace with eclipse.horizon_via_normal?
        elif eclipse_detection == 'simple':
            threshold = kwargs.get('threshold', 185./180.*pi)
            partial = np.abs(self.mesh['mu']) >= threshold
            visible = (self.mesh['mu'] > 0) & -partial
            
            self.mesh['visible'] = visible
            self.mesh['hidden'] = -visible & -partial
            self.mesh['partial'] = partial
        
        # Maybe we don't understand
        else:
            raise ValueError("don't know how to detect eclipses/horizon (set via parameter 'eclipse_detection'")
    
    
    def rotate_and_translate(self,theta=0, incl=0, Omega=0,
              pivot=(0, 0, 0), loc=(0, 0, 0), los=(0, 0, +1), incremental=False,
              subset=None):
        """
        Perform a rotation and translation of a Body.
        """
        # Select a subset (e.g. of partially visible triangles) or not?
        if subset is not None:
            logger.info('rotating subset of mesh into current configuration')
            mesh = self.mesh[subset]
            theta = self.orientation['theta']
            incl = self.orientation['incl']
            Omega = self.orientation['Omega']
            pivot = self.orientation['pivot']
            los = self.orientation['los']
            loc = self.orientation['vector']
        else:
            mesh = self.mesh
        
        # Rotate
        mesh = coordinates.rotate_and_translate(mesh,
                       theta=theta, incl=incl, Omega=Omega,
                       pivot=pivot, los=los, loc=loc, incremental=incremental)
        
        # Replace the subset or not?
        if subset is not None:
            self.mesh[subset] = mesh
        else:
            self.mesh = mesh
        
        # Remember the orientiation, maybe this is useful at some point.
        self.orientation['theta'] = theta
        self.orientation['incl'] = incl
        self.orientation['Omega'] = Omega
        self.orientation['pivot'] = pivot
        self.orientation['los'] = los
        self.orientation['vector'] = loc
    
    
    def rotate(self, theta=0,incl=0,Omega=0,
              pivot=(0,0,0),los=(0,0,+1),incremental=False,
              subset=None):
        """
        Rotate a Body around a pivot point
        """
        return self.rotate_and_translate(theta=theta, incl=incl,
               Omega=Omega, pivot=pivot, los=los, incremental=incremental,
               subset=subset)
    
    
    def translate(self,loc=(0,0,0), los=(0,0,+1),incremental=False, subset=None):
        """
        Translate a Body to another location.
        """
        return self.rotate_and_translate(loc=loc, los=los,
                incremental=incremental, subset=subset)
    
    
    def compute_centers(self):
        """
        Compute the centers of the triangles.
        """
        for idim in range(self.dim):
            self.mesh['_o_center'][:,idim] = self.mesh['_o_triangle'][:,idim::self.dim].sum(axis=1)/3.
            self.mesh['center'][:,idim] = self.mesh['triangle'][:,idim::self.dim].sum(axis=1)/3.
    
    def compute_sizes(self,prefix=''):
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
        # Length of edges
        #side1 = self.mesh[prefix+'triangle'][:,0*self.dim:1*self.dim] -\
                #self.mesh[prefix+'triangle'][:,1*self.dim:2*self.dim]
        #side2 = self.mesh[prefix+'triangle'][:,0*self.dim:1*self.dim] -\
                #self.mesh[prefix+'triangle'][:,2*self.dim:3*self.dim]
        #side3 = self.mesh[prefix+'triangle'][:,1*self.dim:2*self.dim] -\
                #self.mesh[prefix+'triangle'][:,2*self.dim:3*self.dim]
        
        ## Some coefficients
        #a = sqrt(np.sum(side1**2, axis=1))
        #b = sqrt(np.sum(side2**2, axis=1))
        #c = sqrt(np.sum(side3**2, axis=1))
        #k = 0.5 * (a+b+c)
        
        ## And finally the size
        #self.mesh[prefix+'size'] = sqrt( k*(k-a)*(k-b)*(k-c))
        self.mesh[prefix+'size'] = fgeometry.compute_sizes(self.mesh[prefix+'triangle'])
    
    
    def compute_normals(self,prefixes=('','_o_')):
        r"""
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
        # Normal is cross product of two sides
        for prefix in prefixes:
            
            # Compute the sides
            side1 = self.mesh[prefix+'triangle'][:,0*self.dim:1*self.dim] -\
                    self.mesh[prefix+'triangle'][:,1*self.dim:2*self.dim]
            side2 = self.mesh[prefix+'triangle'][:,0*self.dim:1*self.dim] -\
                    self.mesh[prefix+'triangle'][:,2*self.dim:3*self.dim]
            
            # Compute the corss product
            self.mesh[prefix+'normal_'] = -np.cross(side1, side2)


    def area(self):
        """
        Compute the total surface area of a Body.
        
        @return: total surface area in :math:`R^2_\odot`.
        @rtype: float
        """
        return self.mesh['size'].sum()
    
    def get_distance(self):
        globals_parset = self.get_globals()
        if globals_parset is not None:
            distance = globals_parset.request_value('distance', 'Rsol')
        else:
            distance = 10*constants.pc/constants.Rsol
        return distance
    
    def get_coords(self, type='spherical', loc='center', prefix='_o_'):
        """
        Return the coordinates of the Body in a convenient coordinate system.
        
        Phi is longitude (between -PI and +PI)
        theta is colatitude (between 0 and +PI)
        
        Can be useful for surface maps or so.
        
        Needs some work...
        """
        if type == 'spherical':
            index = np.array([1,0,2])
            r1,phi1,theta1 = coordinates.cart2spher_coord(*self.mesh[prefix+'triangle'][:,0:3].T[index])
            r2,phi2,theta2 = coordinates.cart2spher_coord(*self.mesh[prefix+'triangle'][:,3:6].T[index])
            r3,phi3,theta3 = coordinates.cart2spher_coord(*self.mesh[prefix+'triangle'][:,6:9].T[index])
            r4,phi4,theta4 = coordinates.cart2spher_coord(*self.mesh[prefix+'center'].T[index])
            #r = np.hstack([r1,r2,r3,r4])
            #phi = np.hstack([phi1,phi2,phi3,phi4])
            #theta = np.hstack([theta1,theta2,theta3,theta4])
            if loc=='center':
                return r4,phi4,theta4
            else:
                table = np.column_stack([phi1,theta1,r1,phi2,theta2,r2,phi3,theta3,r3])
                return table
        elif type == 'triangulation':
            # Return the coordinates as an array of coordinates and an array
            # of triangle indices
            raise NotImplementedError
            
        else:
            raise ValueError("Don't understand type {}".format(type))
    
    def get_refs(self, category=None, per_category=False,
                 include=('obs','syn','dep')):
        """
        Return the list of unique references.
        
        A *reference* (``ref``) is a unique label linking the right *pbdep*
        with their *syn* and *obs* ParameterSets.
        
        Each reference can appear in three places: the pbdeps, syns and obs,
        and each of them can be optional: if no computations are done, the
        syns are obsolete. When no observations are given (e.g. only
        simulations), no obs is given. When all observables are determined by
        hierarchically packed Bodies (e.g. two stars in a binary system for
        which you want to compute the light curve), then the BodyBag has syn
        and obs but no pbdeps (these are then present in the Star Bodies).
        
        When a ``category`` is given, only the references of that particular
        category will be returned. A category is one of ``rv``, ``lc``, ``sp``...
        
        @param category: category to limit the list of references to.
        @type category: str, recognised category (one of 'lc','rv',...)
        @return: list of unique references (if ``per_category=False``) or a dictionary with unique references per category (if ``per_category=True``)
        @rtype: list of str or dict
        """
        # Collect references and categories
        refs = []
        cats = []
        
        # Make sure the deps are correctly parsed:
        include = list(include)
        if 'pbdep' in include:
            include.append('dep')
        
        # Run over all parameterSets
        for ps in self.walk():
            
            # Possible we encounter "orbit" or another parameterSet that has
            # not been set yet
            if ps is None:
                continue
            
            # Get the context and category: (perhaps we had "lcdep:bla"; this
            # is not implemented yet, but we foresee its future implementation)
            this_context = ps.context.split(':')[0]
            this_category = this_context[:-3]
            
            # Only treat those parameterSets that are deps, syns or obs (i.e.
            # not component, orbit, star...)
            if this_context[-3:] in include:
                
                # Skip nonmatching contexts
                if category is not None and category != this_category:
                    continue
                
                # But add all the rest!
                refs.append(ps['ref'])
                cats.append(this_category)
        
        if category is not None or not per_category:
            # Gaurantee a list of unique references, always given in the same
            # (alphabetical) order.
            return sorted(set(refs))
        
        else:
            
            # Return a dictionary:
            ret_dict = OrderedDict()
            for ref, cat in zip(refs, cats):
            
                # Add the reference if it's not already there
                if cat in ret_dict and not (ref in ret_dict[cat]):
                    ret_dict[cat] = sorted(ret_dict[cat] + [ref])
                
                # Initialize this category if it's not already present
                elif not (cat in ret_dict):
                    ret_dict[cat] = [ref]
            
            return ret_dict
                
    
    def get_parset(self, ref=None, context=None, type='pbdep', category=None):
        """
        Return the parameter set with the given reference from the C{params}
        dictionary attached to the Body.
        
        If reference is None, return the parameterSet of the body itself.
        If reference is an integer, return the "n-th" parameterSet.
        If reference is a string, return the parameterSet with the reference
        matching the string.
        
        If reference is an integer, I cannot figure out if you want the
        pbdep, syn or obs with that reference.  Therefore, you either have
        to give:
        
            - ``context`` (one of ``lcsyn``,``lcdep``,``lcobs``,``rvsyn``...)
            - ``type`` and ``category``, where ``type`` should be ``pbdep``, ``syn`` or ``obs``, and ``category`` should be one of ``lc``,``rv``...
        
        ``context`` has preference over ``type`` and ``category``, i.e. only
        when ``context=None``, the ``type`` and ``category`` will be looked
        at.
        
        returns parset and it's reference (in reverse order)
        
        .. warning::
        
            If possible, please give the context. There is still a bug in the
            code below, that does not iterate over bodies in a lower level if
            nothing was found in this body. Have a look at the first part of the
            code to find the small workaround for that.
        
        """
        # This Body needs to have parameters attached
        if not hasattr(self, 'params'):
            msg = "Requested parset ref={}, type={} but nothing was found"
            logger.info(msg.format(ref, type))
            return None, None
        
        # In some circumstances, we want the parameterSet of the body. This one
        # often also contains information on the atmospheres etc.. but the
        # bolometrically
        if ref is None or ref == '__bol':
            logger.debug("Requested bolometric parameterSet")
            return list(self.params.values())[0], '__bol'
        
        # Next, we check if a context is given. If it is, we immediately can
        # derive the type and category, so there's no need for looping
        if context is not None:
            type = context[-3:]
            
            # dep is an alias for pbdep
            if type == 'dep':
                type = 'pbdep'
            
            # Check if this type and context are available
            if type in self.params and context in self.params[type]:
                # A reference can be a string or an integer, and there are
                # certain number of references
                ref_is_string = isinstance(ref, str)
                n_refs = len(self.params[type][context].values())
                
                # A reference can be a string. If so, check if it exists
                if ref_is_string and ref in self.params[type][context]:
                    ps = self.params[type][context][ref]
                
                # A reference can be an integer. If so, check if it exists 
                elif not ref_is_string and ref < n_refs:
                    ps = self.params[type][context].values()[ref]
                    
                # Look in a level down
                elif hasattr(self, 'bodies'):
                    # We need to iterate over all bodies, and as soon as we
                    # found one, we can return it
                    for body in self.bodies:
                        ret_value = body.get_parset(ref=ref, type=type,
                                             context=context, category=category)
                        if ret_value[0] is not None:
                            return ret_value
                    return None, None
                
                # Else, no luck! We didn't find anything!
                else:
                    logger.debug("Requested parset ref={}, context={} but it does not seem to exist".format(ref,context))
                    return None, None
                
                logger.debug("Requested parset ref={}, context={} and found ref={}, context={}".format(ref,context,ps['ref'],ps.get_context()))
                return ps, ps['ref']
            
            elif hasattr(self, 'bodies'):
                # We need to iterate over all bodies, and as soon as we
                # found one, we can return it    
                for body in self.bodies:
                    ret_value = body.get_parset(ref=ref, type=type,
                                             context=context, category=category)
                    if ret_value[0] is not None:
                        return ret_value
                return None, None
            
            # Elese, no luck! We didn't find anything and really tried our best
            else:
                logger.debug("Requested parset ref={}, context={} but it does not exist".format(ref,context))
                return None, None
            
        # Else, we need to start searching for the right type and category!
        else:
            #counter = 0
            #-- this could be dangerous!!
            if not type in self.params:
                return self[0].get_parset(ref=ref,type=type,context=context,category=category)
            categories = category and [category+type[-3:]] or self.params[type].keys()
            
            for icat in categories:
                if not icat in self.params[type]:
                    # should we throw error here or return None, None?
                    return None, None
                    #raise ValueError('No {} defined: looking in type={}, available={}'.format(icat,type,list(self.params[type].keys())))
                counter = 0
                for ips in self.params[type][icat]:
                    ps = self.params[type][icat][ips]
                    is_ref = ('ref' in ps) and (ps['ref']==ref)
                    is_number = counter==ref
                    if is_ref or is_number:
                        logger.debug("Requested parset ref={}, type={}, category={} and found ref={}, context={}".format(ref,type,category,ps['ref'],ps.get_context()))
                        return ps,ps['ref']
                    counter += 1
            return None,None
    
    def list(self, summary=None, width=79, emphasize=True):
        """
        List with indices all the parameterSets that are available.
        
        Should show some kind of hierarchical list::
        
            entire system
            |
            +==========> Vega+Sirius & B9V (BodyBag placed in orbit)
            |         |
            |         +==========> Vega+Sirius (BodyBag placed in orbit)
            |         |         |
            |         |         +----------> Vega (BinaryRocheStar)
            |         |         |            dep: 2 lcdep, 1 rvdep, 1 spdep, 1
            |         |         |                 ifdep
            |         |         |            syn: 2 lcsyn, 1 rvsyn, 1 spsyn, 1
            |         |         |                 ifsyn
            |         |         |
            |         |         +----------> Sirius (BinaryRocheStar)
            |         |         |            dep: 2 lcdep, 1 rvdep, 1 spdep, 1
            |         |         |                 ifdep
            |         |         |            syn: 2 lcsyn, 1 rvsyn, 1 spsyn, 1
            |         |         |                 ifsyn
            |         |
            |         +==========> B9V_01dd510c-32d8-4a8c-8c1c-38cd92bdd738 (Star placed in orbit)
            |         |            dep: 2 lcdep, 1 rvdep, 1 spdep, 1 ifdep
            |         |            syn: 2 lcsyn, 1 rvsyn, 1 spsyn, 1 ifsyn
            |
            +==========> V380_Cyg (BodyBag placed in orbit)
            |         |
            |         +----------> B1V_primary (BinaryRocheStar)
            |         |            dep: 2 lcdep, 1 rvdep, 1 spdep, 1 ifdep
            |         |            syn: 2 lcsyn, 1 rvsyn, 1 spsyn, 1 ifsyn
            |         |
            |         +----------> B1V_secondary (BinaryRocheStar)
            |         |            dep: 2 lcdep, 1 rvdep, 1 spdep, 1 ifdep
            |         |            syn: 2 lcsyn, 1 rvsyn, 1 spsyn, 1 ifsyn                    
        
        Possibilities:
        
            - :envvar:`summary=None`: returns only the structure of the system,
              but no information on obserations or parameters::
              
                23fc64ba-257c-4278-b05a-b307d164bc5b (BodyBag)
                |
                +----------> primary (BinaryRocheStar)
                |
                +----------> secondary (BinaryRocheStar)
              
            - :envvar:`summary='short'`: only lists information on how many
              of each type (obs, syn and dep) are attached to the system::
              
                23fc64ba-257c-4278-b05a-b307d164bc5b (BodyBag)
                obs: 4 lcobs
                syn: 4 lcsyn
                |
                +----------> primary (BinaryRocheStar)
                |            dep: 4 lcdep, 1 rvdep
                |            obs: 0 lcobs, 1 rvobs
                |            syn: 4 lcsyn, 1 rvsyn
                |
                +----------> secondary (BinaryRocheStar)
                |            dep: 4 lcdep, 1 rvdep
                |            obs: 0 lcobs, 1 rvobs
                |            syn: 4 lcsyn, 1 rvsyn
                
            - :envvar:`summary='long'`: lists the obs and dep attached, together
              with their reference strings::
              
                23fc64ba-257c-4278-b05a-b307d164bc5b (BodyBag)
                lcobs: stromgren.u, stromgren.b, stromgren.v, stromgren.y
                |
                +----------> primary (BinaryRocheStar)
                |            lcdep: stromgren.u, stromgren.b, stromgren.v, stromgren.y
                |            rvdep: rv1
                |            rvobs: rv1
                |
                +----------> secondary (BinaryRocheStar)
                |            lcdep: stromgren.u, stromgren.b, stromgren.v, stromgren.y
                |            rvdep: rv2
                |            rvobs: rv2
        
            - :envvar:`summary='physical'`: lists only the physical parameters,
               but not the datasets::
                
                V380Cyg (BodyBag)
                |
                +----------> B1V_primary (BinaryRocheStar)
                |            component: atm=blackbody, gravblaw=zeipel, alb=1.000000,
                |                   redist=0.000000, redisth=0.000000, syncpar=1.541860,
                |                   gravb=1.000000, pot=4.700000, teff=21700 K,
                |                   morphology=unconstrained, irradiator=True, abun=0.000000,
                |                   label=B1V_primary, ld_func=linear, ld_coeffs=[0.5]
                |            orbit: dpdt=0.000000 s/yr, dperdt=0.000000 deg/yr, ecc=0.226100,
                |                   t0=2454000.000000 JD, t0type=periastron passage,
                |                   incl=78.783591 deg, label=V380Cyg, period=12.425719 d,
                |                   per0=141.480000 deg, phshift=0.000000, q=0.609661,
                |                   sma=60.252837 Rsol, long_an=0.000000 deg,
                |                   c1label=B1V_primary, c2label=B1V_secondary
                |            mesh: delta=0.100000, maxpoints=20000, alg=c
                |            reddening: law=fitzpatrick2004, extinction=0.000000,
                |                   passband=JOHNSON.V, Rv=3.100000
                |
                +----------> B1V_secondary (BinaryRocheStar)
                |            component: atm=blackbody, gravblaw=zeipel, alb=1.000000,
                |                   redist=0.000000, redisth=0.000000, syncpar=2.501978,
                |                   gravb=1.000000, pot=10.602351, teff=22400 K,
                |                   morphology=unconstrained, irradiator=True, abun=0.000000,
                |                   label=B1V_secondary, ld_func=linear, ld_coeffs=[0.5]
                |            orbit: dpdt=0.000000 s/yr, dperdt=0.000000 deg/yr, ecc=0.226100,
                |                   t0=2454000.000000 JD, t0type=periastron passage,
                |                   incl=78.783591 deg, label=V380Cyg, period=12.425719 d,
                |                   per0=141.480000 deg, phshift=0.000000, q=0.609661,
                |                   sma=60.252837 Rsol, long_an=0.000000 deg,
                |                   c1label=B1V_primary, c2label=B1V_secondary
                |            mesh: delta=0.100000, maxpoints=20000, alg=c
                |            reddening: law=fitzpatrick2004, extinction=0.000000,
                |                   passband=JOHNSON.V, Rv=3.100000                  
               
            - :envvar:`summary='full'`: lists all parameters and datasets
        
        @param emphasize: bring structure in the text via boldfacing
        @type emphasize: bool
        @param summary: type of summary (envvar:`long`, envvar:`short`, envvar:`physical`, envvar:`full`)
        @type summary: str
        """
        # Make sure to not print out all array variables
        old_threshold = np.get_printoptions()['threshold']
        np.set_printoptions(threshold=8)
        
        def add_summary_short(thing, text, width=79):
            """
            Add information on pbdeps, obs and syn
            """
            
            # Construct the  "|     |     " string that preceeds the summary info
            # We need to have the same length as the previous line. If there
            # is no previous line, or it is not indented, we don't need to indent
            # this one
            try:
                indent = text[-1].split('+')[0] \
                        + '|'\
                        + ' '*len(text[-1].split('+')[1].split('>')[0]) \
                        + '  '
            except IndexError:
                indent = ''
                
            # Collect info on the three types    
            summary = []
            for ptype in ['pbdep', 'obs', 'syn']:
                
                # Which type is this? keep track of the number of members, if
                # there are none, don't report
                this_type = ['{}: '.format(ptype[-3:])]
                ns = 0
                # Loop over all categories and make a string
                for category in ['lc', 'rv', 'sp', 'if', 'pl', 'etv', 'am']:
                    lbl = (category+ptype[-3:])
                    if ptype in thing.params and lbl in thing.params[ptype]:
                        this_type.append('{} {}'.format(len(thing.params[ptype][lbl]),lbl))
                        ns += len(thing.params[ptype][lbl])
                
                # Only report if there are some
                if ns > 0:
                    mystring = this_type[0]+', '.join(this_type[1:])
                    summary.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+5*' ')))
            
            text += summary
        
        
        def add_summary_long(thing, text, width=79):
            """
            Add information on pbdeps, obs and syn
            """
           
            # Construct the  "|     |     " string that preceeds the summary info
            # We need to have the same length as the previous line. If there
            # is no previous line, or it is not indented, we don't need to indent
            # this one
            try:
                indent = text[-1].split('+')[0] \
                       + '|'\
                       + ' '*len(text[-1].split('+')[1].split('>')[0]) \
                       + '  '
            except IndexError:
                indent = ''
               
            # Collect references
            summary = []
            # Loop over all categories and make a string
            for category in ['lc', 'rv', 'sp', 'if', 'pl', 'etv', 'am']:
               
                for ptype in ['pbdep','obs']:
                    ns = 0 
                    lbl = (category+ptype[-3:])
                    mystring = ['{}: '.format(lbl)]
                    if ptype in thing.params and lbl in thing.params[ptype]:
                        for ref in thing.params[ptype][lbl]:
                            mystring.append(ref)
                        ns += len(thing.params[ptype][lbl])
                    mystring = mystring[0] + ', '.join(mystring[1:])
                    # Only report if there are some
                    if ns > 0:
                        summary.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+7*' ', width=79)))
           
            text += summary
        
        def add_summary_cursory(thing, text, width=79):
            """
            Add information on pbdeps, obs and syn
            """
           
            # Construct the  "|     |     " string that preceeds the summary info
            # We need to have the same length as the previous line. If there
            # is no previous line, or it is not indented, we don't need to indent
            # this one
            try:
                indent = text[-1].split('+')[0] \
                       + '|'\
                       + ' '*len(text[-1].split('+')[1].split('>')[0]) \
                       + '  '
            except IndexError:
                indent = ''
               
            # Collect references
            summary = []
            
            mystring = []
            # Loop over all stuff but just list the available contexts
            # If this thing has a child, see if it has an orbit and print it
            if hasattr(thing, 'bodies') and 'orbit' in thing.get_children()[0].params:
                i_have_orbit = ['orbit']
            else:
                i_have_orbit = []
            
            # Global system parameters
            for param in thing.params:
                if param in ['pbdep', 'obs', 'syn']:
                    continue
                
                if param == 'orbit':
                    continue
                
                mystring.append(param)
                
                if param == 'globals':
                    mystring += i_have_orbit
                    i_have_orbit = []
            
            # Add orbit in any case
            mystring += i_have_orbit
            summary = []
            for imystring in mystring:
                summary.append("\n".join(textwrap.wrap(imystring, initial_indent=indent, subsequent_indent=indent+7*' ', width=width)))

            # Loop over all categories and make a string
            for category in ['lc', 'rv', 'sp', 'if', 'pl', 'etv', 'am']:
               
                # first pbdep
                ptype = 'pbdep'
                ns = 0 
                lbl = (category+ptype[-3:])
                mystring = [italicize('{}: '.format(lbl))]
                if ptype in thing.params and lbl in thing.params[ptype]:
                    for ref in thing.params[ptype][lbl]:
                        mystring.append(ref)
                    ns += len(thing.params[ptype][lbl])
                mystring = mystring[0] + ', '.join(mystring[1:])
                # Only report if there are some
                if ns > 0:
                    summary.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+7*' ', width=79)))
                
                # then obs, if there are any
                ptype = 'obs'
                ns = 0 
                lbl = (category+ptype[-3:])
                mystring = [italicize('{}: '.format(lbl))]
                mystring2 = [italicize('{}: '.format(category+'syn'))]
                if ptype in thing.params and lbl in thing.params[ptype]:
                    for ref in thing.params[ptype][lbl]:
                        mystring.append(ref)
                        mystring2.append(ref)
                        
                        if thing.params[ptype][lbl][ref].get_enabled()==False:
                            mystring[-1] = strikethrough(mystring[-1])
                    ns += len(thing.params[ptype][lbl])
                mystring = mystring[0] + ', '.join(mystring[1:])
                mystring2 = mystring2[0] + ', '.join(mystring2[1:])
                # Only report if there are some
                if ns > 0:
                    summary.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+7*' ', width=79)))
                    summary.append("\n".join(textwrap.wrap(mystring2, initial_indent=indent, subsequent_indent=indent+7*' ', width=79)))

                    
                
            text += summary
        
        
        
        def add_summary_physical(thing, text, width=79):
            """
            Add information on pbdeps, obs and syn
            """
           
            # Construct the  "|     |     " string that preceeds the summary info
            # We need to have the same length as the previous line. If there
            # is no previous line, or it is not indented, we don't need to indent
            # this one
            try:
                indent = text[-1].split('+')[0] \
                       + '|'\
                       + ' '*len(text[-1].split('+')[1].split('>')[0]) \
                       + '  '
            except IndexError:
                indent = ''
               
            # Collect references
            summary = []
            # Loop over all categories and make a string
            for param in thing.params:
                if param in ['pbdep', 'obs', 'syn']:
                    continue
                lbl = param
                mystring = ['{}: '.format(lbl)]
                if not isinstance(thing.params[param],list):
                    iterover = [thing.params[param]]
                else:
                    iterover = thing.params[param]
                for iiterover in iterover:
                    for par in iiterover:
                        mystring.append("{}={}".format(par,iiterover.get_parameter(par).to_str()))
                        if iiterover.get_parameter(par).has_unit():
                            mystring[-1] += ' {}'.format(iiterover.get_parameter(par).get_unit())
                        if emphasize and iiterover.get_parameter(par).get_adjust():
                            mystring[-1] = "\033[32m" + mystring[-1] +  '\033[m'
                mystring = mystring[0] + ', '.join(mystring[1:])
                summary.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+7*' ', width=width)))

            text += summary
        
        def add_summary_full(thing, text, width=79):
            """
            Add information on pbdeps, obs and syn
            """
            # Construct the  "|     |     " string that preceeds the summary info
            # We need to have the same length as the previous line. If there
            # is no previous line, or it is not indented, we don't need to indent
            # this one
            try:
                indent = text[-1].split('+')[0] \
                       + '|'\
                       + ' '*(len(text[-1].split('+-')[1].split('>')[0])+1) \
                       + '  '
            except IndexError:
                indent = ''
               
            # Collect references
            summary = []
            # If this thing has a child, see if it has an orbit and print it
            i_have_orbit = []
            if hasattr(thing, 'bodies') and 'orbit' in thing.get_children()[0].params:
                lbl = 'orbit'
                mystring = ['{}: '.format(lbl)]
                iiterover = thing.get_children()[0].params['orbit']
                for par in iiterover:
                        mystring.append("{}={}".format(par,iiterover.get_parameter(par).to_str()))
                        if iiterover.get_parameter(par).has_unit():
                            mystring[-1] += ' {}'.format(iiterover.get_parameter(par).get_unit())
                        if emphasize and iiterover.get_parameter(par).get_adjust():
                            mystring[-1] = "\033[32m" + mystring[-1] +  '\033[m'
                mystring = mystring[0] + ', '.join(mystring[1:])
                i_have_orbit.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+7*' ', width=width)))
            
            # Loop over all categories and make a string
            for param in thing.params:
                if param in ['pbdep', 'obs', 'syn']:
                    continue
                
                # Skip orbits, we should've taken care of those at a higher level
                if param == 'orbit':
                    continue
                
                lbl = param
                mystring = ['{}: '.format(lbl)]
                if not isinstance(thing.params[param],list):
                    iterover = [thing.params[param]]
                else:
                    iterover = thing.params[param]
                for iiterover in iterover:
                    for par in iiterover:
                        mystring.append("{}={}".format(par,iiterover.get_parameter(par).to_str()))
                        if iiterover.get_parameter(par).has_unit():
                            mystring[-1] += ' {}'.format(iiterover.get_parameter(par).get_unit())
                        if emphasize and iiterover.get_parameter(par).get_adjust():
                            mystring[-1] = "\033[32m" + mystring[-1] +  '\033[m'
                mystring = mystring[0] + ', '.join(mystring[1:])
                summary.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+7*' ', width=width)))
                
                # we need to insert orbit after globals if there is any (seriously)
                if lbl == 'globals':
                    summary += i_have_orbit
                    i_have_orbit = []
            
            # if there were no globals, add the orbit
            summary += i_have_orbit
            
            # Loop over all pbdep and make a string
            for param in thing.params:
                if not param == 'pbdep':
                    continue

                for ptype in thing.params[param]:
                    for iref,ref in enumerate(thing.params[param][ptype]):
                        
                        # The pbdep
                        lbl = '{}[{}]'.format(ptype, iref)
                        mystring = ['{}'.format(lbl)]
                        iterover = thing.params[param][ptype][ref].keys()
                        iiterover = thing.params[param][ptype][ref]
                        # First the reference for clarity:
                        if 'ref' in iterover:
                            iterover.remove('ref')
                            iterover = ['ref'] + iterover
                        
                        mystring[-1] += ' ({}@{}@{}): '.format(ref,ptype,thing.get_label())
                        
                        for par in iterover:
                            mystring.append("{}={}".format(par,iiterover.get_parameter(par).to_str()))
                            if iiterover.get_parameter(par).has_unit():
                                mystring[-1] += ' {}'.format(iiterover.get_parameter(par).get_unit())
                            if emphasize and iiterover.get_parameter(par).get_adjust():
                                mystring[-1] = "\033[32m" + mystring[-1] +  '\033[m'
                        
                        mystring = mystring[0] + ', '.join(mystring[1:])
                        summary.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+7*' ', width=width)))
                        
                        # Is there an obs here?
                        ptype_ = ptype[:-3]+'obs'
                        if not ptype_ in thing.params['obs']:
                            continue
                        if not ref in thing.params['obs'][ptype_]:
                            continue
                        lbl = '{}[{}]'.format(ptype_, iref)
                        mystring = ['{}'.format(lbl)]
                        oparam = 'obs'
                        iterover = thing.params[oparam][ptype_][ref].keys()
                        iiterover = thing.params[oparam][ptype_][ref]
                        
                        mystring[-1] += ' ({}@{}@{}, n={}): '.format(ref,ptype[:-3]+'obs',thing.get_label(),len(iiterover))
                        
                        # First the reference for clarity:
                        if 'ref' in iterover:
                            iterover.remove('ref')
                            iterover = ['ref'] + iterover
                        
                        for par in iterover:
                            mystring.append("{}={}".format(par,iiterover.get_parameter(par).to_str()))
                            if iiterover.get_parameter(par).has_unit():
                                mystring[-1] += ' {}'.format(iiterover.get_parameter(par).get_unit())
                            if emphasize and iiterover.get_parameter(par).get_adjust():
                                mystring[-1] = "\033[32m" + mystring[-1] +  '\033[m'
                        
                        mystring = mystring[0] + ', '.join(mystring[1:])
                        
                        summary.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+7*' ', width=width)))
                        
                        # The syn should match the obs
                        ptype_ = ptype[:-3]+'syn'
                        lbl = '{}[{}]'.format(ptype_, iref)
                        mystring = ['{}'.format(lbl)]
                        oparam = 'syn'
                        iiterover = thing.get_synthetic(category=ptype[:-3], ref=ref)
                        iterover = iiterover.keys()
                        
                        mystring[-1] += ' ({}@{}@{}, n={}): '.format(ref,ptype_,thing.get_label(),len(iiterover))
                        
                        # First the reference for clarity:
                        if 'ref' in iterover:
                            iterover.remove('ref')
                            iterover = ['ref'] + iterover
                        
                        for par in iterover:
                            mystring.append("{}={}".format(par,iiterover.get_parameter(par).to_str()))
                            if iiterover.get_parameter(par).has_unit():
                                mystring[-1] += ' {}'.format(iiterover.get_parameter(par).get_unit())
                            if emphasize and iiterover.get_parameter(par).get_adjust():
                                mystring[-1] = "\033[32m" + mystring[-1] +  '\033[m'
                        
                        mystring = mystring[0] + ', '.join(mystring[1:])
                        summary.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+7*' ', width=width)))
                        
            # Loop over all pbdep and make a string
            for param in thing.params:
                if not param == 'obs':
                    continue
                
                for ptype in thing.params[param]:
                    for iref,ref in enumerate(thing.params[param][ptype]):
                        
                        if 'pbdep' in thing.params and ptype[:-3]+'dep' in thing.params['pbdep'] and ref in thing.params['pbdep'][ptype[:-3]+'dep']:
                            continue

                        lbl = '{}[{}]'.format(ptype, iref)
                        mystring = ['{}'.format(lbl)]
                        iterover = thing.params[param][ptype][ref].keys()
                        iiterover = thing.params[param][ptype][ref]
                        # First the reference for clarity:
                        if 'ref' in iterover:
                            iterover.remove('ref')
                            iterover = ['ref'] + iterover
                        mystring[-1] += ' ({}@{}@{}, n={}): '.format(ref,ptype[:-3]+'obs',thing.get_label(),len(iiterover))
                        for par in iterover:
                            mystring.append("{}={}".format(par,iiterover.get_parameter(par).to_str()))
                            if iiterover.get_parameter(par).has_unit():
                                mystring[-1] += ' {}'.format(iiterover.get_parameter(par).get_unit())
                            if emphasize and iiterover.get_parameter(par).get_adjust():
                                mystring[-1] = "\033[32m" + mystring[-1] +  '\033[m'
                        
                        mystring = mystring[0] + ', '.join(mystring[1:])
                        if iiterover.get_enabled()==False:
                            mystring = strikethrough(mystring)
                        summary.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+7*' ', width=width)))
                        
                        lbl = '{}[{}]'.format(ptype[:-3]+'syn', iref)
                        mystring = ['{}'.format(lbl)]
                        iiterover = thing.get_synthetic(category=ptype[:-3], ref=ref).asarray()
                        iterover = iiterover.keys()
                        
                        # First the reference for clarity:
                        if 'ref' in iterover:
                            iterover.remove('ref')
                            iterover = ['ref'] + iterover
                        mystring[-1] += ' ({}@{}@{}, n={}): '.format(ref,ptype[:-3]+'syn',thing.get_label(),len(iiterover))
                        for par in iterover:
                            mystring.append("{}={}".format(par,iiterover.get_parameter(par).to_str()))
                            if iiterover.get_parameter(par).has_unit():
                                mystring[-1] += ' {}'.format(iiterover.get_parameter(par).get_unit())
                            if emphasize and iiterover.get_parameter(par).get_adjust():
                                mystring[-1] = "\033[32m" + mystring[-1] +  '\033[m'
                        
                        mystring = mystring[0] + ', '.join(mystring[1:])
                        summary.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+7*' ', width=width)))

            text += summary
        
        if emphasize:
            def emphasize(text):
                return '\033[1m\033[4m' + text + '\033[m'
            def italicize(text):
                return '\x1B[3m' + text + '\033[m'
            def strikethrough(text):
                return '\x1b[9m' + text + '\x1b[29m'
        else:
            emphasize = lambda x: x
            italicize = lambda x: x
            strikethrough = lambda x:x
        
        if summary:
            add_summary = locals()['add_summary_'+summary]
        else:
            add_summary = lambda thing, text, width: ''
        
        # Top level string: possible the BB has no label
        try:   
            text = ["{} ({})".format(self.get_label(),self.__class__.__name__)]
        except ValueError:
            text = ["<nolabel> ({})".format(self.__class__.__name__)]
        
        text[-1] = emphasize(text[-1])
        
        # Add the summary of the top level thing
        add_summary(self, text, width)

        # Keep track of the previous label, we want to skip indentation for
        # single-body BodyBags (that is a Body(Bag) in a BodyBag just for
        # orbit-purposes)
        previous_label = text[-1]
        
        # Walk over all things in the system
        for loc, thing in self.walk_all():
            # Iteration happens over all ParameterSets and bodies. We are only
            # interested in the Bodies.
            if not isinstance(thing, Body):
                continue
            
            # Get the label from this thing
            try:
                label = thing.get_label()
            except ValueError:
                label = '<nolabel>'
                
            label = emphasize(label)
            
            # Get the parent to report if the system is connected/disconnected
            parent = thing.get_parent()
            if parent is not None and not parent.connected:
                label += ' (disconnected)'
            
                
            # If this thing is a BodyBag, treat it as such
            if isinstance(thing, BodyBag) and previous_label != label:
                
                # A body bag is represented by a "|   |    +=======>" sign
                level = len(set(loc))-1 
                prefix = ('|'+(' '*9))*(level-1) ### was level-2 but line above was not "set"
                text.append(prefix + '|')
                prefix += '+' + '='*10
                text.append(prefix + '> ' + label)
                
                # But avoid repetition
                if loc[-1]==loc[-2]:
                    continue
            
            # If the label is the same as the previous one and it's a BodyBag
            # report that it is just one that is used to place something in an
            # orbit
            elif isinstance(thing, BodyBag):
                text[-1] += ' (BodyBag placed in orbit)'
            
            # If this label is the same as the previous one, but it's not a
            # BodyBag, report so.
            elif previous_label == label:
                text[-1] += ' ({} placed in orbit)'.format(thing.__class__.__name__)
            
            # Else report the normal Body
            elif previous_label != label:
                
                # A normal body is represented by a "|   |    +-------->" sign
                level = len(set(loc))-1
                prefix = ('|'+(' '*9))*(level-1)
                if level > 0:
                    text.append(prefix + '|')
                    prefix += '+' + '-'*10
                    text.append(prefix + '> ' + label + ' ({})'.format(thing.__class__.__name__))
            
            # Add a summary for this body
            add_summary(thing, text, width)
            
            # Remember the label
            previous_label = label
            
        # Default printoption
        np.set_printoptions(threshold=old_threshold)    
            
        return "\n".join(text)
    
    def summary(self):
        return self.list(summary='cursory')
    
    def tree(self):
        return self.list(summary='full')
    
    def clear_synthetic(self):
        """
        Clear the body from all calculated results.
        """
        #result_sets = dict(lcsyn=datasets.LCDataSet,
                       #rvsyn=datasets.RVDataSet,
                       #spsyn=datasets.SPDataSet,
                       #ifsyn=datasets.IFDataSet,
                       #plsyn=datasets.PLDataSet)
        if hasattr(self,'params') and 'syn' in self.params:
            for pbdeptype in self.params['syn']:
                for ref in self.params['syn'][pbdeptype]:
                    old = self.params['syn'][pbdeptype][ref]
                    # previously we re initialized the whole set
                    #new = result_sets[pbdeptype](context=old.context,ref=old['ref'])
                    # now we just reset every variable
                    old.clear()
                    #self.params['syn'][pbdeptype][ref] = old
                    check =True
                
        #logger.info('Removed previous synthetic calculations where present')
        
        
    def get_synthetic(self, category=None, ref=0, cumulative=True):
        """
        Retrieve results from synethetic calculations.
        
        If the C{ref} is not present in the C{syn} section of the C{params}
        attribute of this Body, then C{None} will be returned.
        
        @param type: type of synthetics to retrieve
        @type type: str, one of C{lcsyn}, C{rvsyn}, C{spsyn}, C{ifsyn}
        @param ref: reference to the synthetics
        @type ref: string (the reference) or integer (index)
        @param cumulative: sum results over all lower level Bodies
        @type cumulative: bool
        @return: the ParameterSet containing the synthetic calculations
        @rtype: ParameterSet
        """
        base, ref = self.get_parset(ref=ref, type='syn', category=category)
        
        return base

    
    def add_obs(self, obs):
        """
        Add a list of DataSets to the Body.
        
        @param obs: list of DataSets
        @type obs: list
        """
        # Add data to params
        parsed_refs = _parse_obs(self, obs)
        logger.info('added obs {0}'.format(parsed_refs))
        return parsed_refs
    
    
    def remove_obs(self, refs):
        """
        Remove observation (and synthetic) ParameterSets from the Body.
        """
        # Seems logical that a user would give a string
        if isinstance(refs, str):
            refs = [refs]
        
        # Make the refs unique, cycle over them and remove them
        refs = set(refs)
        
        for dep in self.params['obs']:
            
            syn = dep[:-3] + 'syn'
            keys = set(self.params['obs'][dep].keys())
            intersect = list(keys & refs)
            
            # As long as there are references in common with the dictionary,
            # remove one
            while intersect:
                ref = intersect.pop()
                self.params['obs'][dep].pop(ref)
                if syn in self.params['syn'] and ref in self.params['syn'][syn]:
                    self.params['syn'][syn].pop(ref)
 
                # Drop fields
                fields = 'ld_{0}'.format(ref),\
                         'lproj_{0}'.format(ref),\
                         'velo_{0}'.format(ref),\
                         '_o_velo_{0}'.format(ref)
                self.mesh = pl.mlab.rec_drop_fields(self.mesh, fields)
                logger.info('removed obs {0}'.format(ref))
    
    
    def __add__(self, other):
        """
        Combine two bodies in a BodyBag.
        """
        return BodyBag([self, other])
    #}
    
    
    #{ Input and output
    def plot2D(self, **kwargs):
        """
        Plot mesh in 2D using matplotlib.
        
        For more information, see :py:func:`phoebe.backend.observatory.image`.
        """
        return observatory.image(self, **kwargs)
        
        
    def plot3D(self,select=None,normals=False,scalars=None,centers=False,
                  velos=False,B=False,offset=(0,0,0),savefig=False,coframe='',**kwargs):
        """
        Plot mesh in 3D using Mayavi.
        
        set e.g. select='teff'
        
        or also 'intensity','rv','logg','mu' etc..
        
        intensity equals ld[:,4]
        
        Possible keys for selecting points:
        'partial','hidden','visible'
        
        Ein bisschen mehr information about the defaults:
            - for rv: colormap is reversed RedBlue (Red moving away, Blue coming
              towards). The default lower and upper limits are (-200/+200 km/s).
              The units are km/s, i.e. the values from the mesh are converted from
              Rsol/d to km/s.
             
        A particularly interesting extra keyword argument is :envvar:`representation='wireframe'`,
        which allows you to see through the Body.
            
        For more information on kwargs, see the mayavi documentation.
        """        
        if not enable_mayavi:
            print("I told you before: Mayavi is not installed. Call to plot3D is ignored.")
            return None
        if select=='rv':
            kwargs.setdefault('colormap','RdBu')
        if select=='teff':
            kwargs.setdefault('colormap','hot')
        #kwargs.setdefault('representation','wireframe')
        #kwargs.setdefault('tube_radius',None)
        if select is None:
            select = ['hidden','visible','partial']
        elif isinstance(select,str):
            if select in self._plot['plot3D']:
                kwargs.setdefault('vmin',self._plot['plot3D'][select][0])
                kwargs.setdefault('vmax',self._plot['plot3D'][select][1])
            select = [select]
        else:
            select = [select]
            
        if savefig:
            mlab.figure(bgcolor=(0.5,0.5,0.5),size=kwargs.pop('size',(600,600)))
        
        scale_factor = kwargs.pop('scale_factor',1.0)
        
        for si,keep in enumerate(select):
            kwargs_ = kwargs.copy()
            if keep=='hidden':
                kwargs_.setdefault('color',(1.0,0.0,0.0))
            elif keep=='partial':
                #kwargs_.setdefault('color',(0.0,1.0,0.0))
                kwargs_.setdefault('color',(0.0,0.0,1.0))
            elif keep=='visible':
                kwargs_.setdefault('color',(0.0,1.0,0.0))
            elif keep=='intensity':
                kwargs_['scalars'] = np.hstack([self.mesh['ld___bol'][:,4] for i in range(3)])
            elif keep=='rv':
                kwargs_['scalars'] = np.hstack([self.mesh['velo___bol_'][:,2] for i in range(3)])
                kwargs_['scalars'] = conversions.convert('Rsol/d','km/s',kwargs_['scalars'])
            elif keep=='Bx':
                kwargs_['scalars'] = np.hstack([self.mesh['B_'][:,0] for i in range(3)])
            elif keep=='By':
                kwargs_['scalars'] = np.hstack([self.mesh['B_'][:,1] for i in range(3)])
            elif keep=='Bz':
                kwargs_['scalars'] = np.hstack([self.mesh['B_'][:,2] for i in range(3)])
            elif keep=='sphercoord_long':
                kwargs_['scalars'] = np.hstack([coordinates.cart2spher_coord(*self.mesh['_o_center'].T)[1] for i in range(3)])
            elif keep=='sphercoord_lat':
                kwargs_['scalars'] = np.hstack([coordinates.cart2spher_coord(*self.mesh['_o_center'].T)[2] for i in range(3)])
            elif isinstance(keep,str):
                kwargs_['scalars'] = np.hstack([self.mesh[keep] for i in range(3)])
                keep = None
            else:
                kwargs_['scalars'] = np.hstack([keep for i in range(3)])
                keep = None
            
            #-- make a subselection on the centers to plot
            if isinstance(keep,str) and keep in self.mesh.dtype.names:
                keep = self.mesh[keep]
            elif isinstance(keep,str) and keep[:2]=='in' and keep[2:] in self.mesh.dtype.names:
                keep = -self.mesh[keep[2:]]
            else:
                keep = np.array(np.ones(len(self.mesh)),bool)
            if not keep.sum():
                logger.warning('%s not available to plot (due to configuration or unset time)'%(select[si]))
                continue
            N = self.mesh[coframe+'triangle'][keep].shape[0]
            x = np.hstack([self.mesh[coframe+'triangle'][keep,i] for i in range(0,9,3)])
            y = np.hstack([self.mesh[coframe+'triangle'][keep,i] for i in range(1,9,3)])
            z = np.hstack([self.mesh[coframe+'triangle'][keep,i] for i in range(2,9,3)])
            triangles = [(i,N+i,2*N+i) for i in range(N)]
            x0,y0,z0 = offset
            #-- plot the normals as arrows in the same color as the triangles
            
            if normals:
                x1,y1,z1 = self.mesh[coframe+'center'][keep].T
                u1,v1,w1 = self.mesh[coframe+'normal_'][keep].T
                mlab.quiver3d(x1+x0,y1+y0,z1+z0,u1,v1,w1,color=kwargs_.get('color'),scale_factor=scale_factor,scale_mode='none')
            if velos:
                x1,y1,z1 = self.mesh[coframe+'center'][keep].T
                u1,v1,w1 = self.mesh[coframe+'velo___bol_'][keep].T
                mlab.quiver3d(x1+x0,y1+y0,z1+z0,u1,v1,w1,color=kwargs_.get('color'),scale_factor=scale_factor,scale_mode='none')
            if B:
                x1,y1,z1 = self.mesh[coframe+'center'][keep].T
                u1,v1,w1 = self.mesh[coframe+'B_'][keep].T
                mlab.quiver3d(x1+x0,y1+y0,z1+z0,u1,v1,w1,color=kwargs_.get('color'),scale_factor=scale_factor,scale_mode='none')
            #-- what color to use?
            if isinstance(scalars,str):
                thrash = kwargs_.pop('color')
                kwargs_['scalars'] = np.hstack([self.mesh[scalars][keep] for i in range(3)])
            
            #-- and plot!
            if len(x):
                trimesh = mlab.triangular_mesh(x+x0,y+y0,z+z0,triangles,**kwargs_)
                if select[si]=='rv': # reverse colormap
                    lut = trimesh.module_manager.scalar_lut_manager.lut.table.to_array()
                    trimesh.module_manager.scalar_lut_manager.lut.table = lut[::-1]
            if centers:
                mlab.points3d(self.mesh[coframe+'center'][:,0],
                              self.mesh[coframe+'center'][:,1],
                              self.mesh[coframe+'center'][:,2],
                              color=(0.9,0.9,0.9),
                              scale_factor=scale_factor,scale_mode='none')
        #-- set to parallel projection, and set line-of-sight in X-direction
        mlab.gcf().scene.parallel_projection = True
        mlab.gcf().scene.z_plus_view()
        if savefig:
            mlab.colorbar()
            mlab.savefig(savefig)
            mlab.close()
        #mlab.view(focalpoint=)
    
    
    def save(self, filename):
        """
        Save a class to an file.
        
        You need to purge signals before writing.
        """
        ff = open(filename, 'w')
        pickle.dump(self, ff)
        ff.close()  
        logger.info('Saved model to file {} (pickle)'.format(filename))
    
    def copy(self):
        """
        Copy this instance.
        """
        return copy.deepcopy(self)
                            
    @decorators.parse_ref
    def etv(self,ref='alletvdep',times=None,ltt=False):
        """
        Compute eclipse timings and add results to the pbdep ParameterSet.
        
        """
        #-- don't bother if we cannot do anything...
        if hasattr(self,'params') and 'obs' in self.params and 'orbit' in self.params:
            orbit = self.params['orbit']
            
            sibling = self.get_sibling()
           
            orbits, components = self.get_orbits() 
            
            for lbl in ref:
                etvsyn,lbl = self.get_parset(type='syn',ref=lbl)
                etvobs,lbl = self.get_parset(type='obs',ref=lbl)
                
                if etvsyn is (None, None) or etvobs is (None, None):
                    continue
                
                if times is None: # then default to what is in the etvobs
                    times = etvobs['time'] # eventually change to compute from cycle number
               
                # get true observed times of eclipse (with LTTE, etc)
                objs, vels, t = keplerorbit.get_barycentric_hierarchical_orbit(times, orbits, components, barycentric=ltt)
                
                # append to etvsyn
                etvsyn['time'] = np.append(etvsyn['time'],times)
                etvsyn['eclipse_time'] = np.append(etvsyn['eclipse_time'],t)
                etvsyn['etv'] = np.append(etvsyn['etv'],t-times)
           

    
    @decorators.parse_ref
    def ifm(self, ref='allifdep', time=None, obs=None, correct_oversampling=1,
           beaming_alg='none', save_result=True):
        """
        Compute interferometry.
        
        You can only do this if you have observations attached.
        
        For details on the computations, see :py:func:`observatory.ifm <phoebe.backend.observatory.ifm>`
        """
        # We need to get the observation ParameterSet so that we know all the
        # required info on **what** exactly to compute (**how** to compute it
        # is contained in the pbdep)
        if obs is None and hasattr(self,'params') and 'obs' in self.params \
                                    and 'ifobs' in self.params['obs']:
            # Compute the IFM profiles for all references
            for lbl in ref:
                # Get the observations if they are not given already
                if obs is None:
                    ifobs, lbl = self.get_parset(type='obs', ref=lbl)
                    if lbl is None:
                        continue
                else:
                    ifobs = obs
                
                # Compute the IFM profiles for this reference
                self.ifm(ref=lbl, time=time, obs=ifobs,
                         correct_oversampling=correct_oversampling,
                         beaming_alg=beaming_alg, save_result=save_result)
        
        # If no obs are given and there are no obs attached, assume we're a
        # BodyBag and descend into ourselves:
        elif obs is None:
            try:
                for body in self.bodies:
                    body.ifm(ref=ref, time=time, obs=obs,
                         correct_oversampling=correct_oversampling,
                         beaming_alg=beaming_alg, save_result=save_result)
            except AttributeError:
                pass
        
        # If obs are given, there is no need to look for the references, and we
        # can readily compute the visibilities
        else:
            ref = obs['ref']
            
            # Well, that is, we will only compute the IFM if there are pbdeps
            # on this body. If not, we assume it's a BodyBag and descend into
            # the bodies.
            if not (hasattr(self,'params') and 'pbdep' in self.params \
                                    and 'ifdep' in self.params['pbdep']):
                try:
                    for body in self.bodies:
                        body.ifm(ref=ref, time=time, obs=obs,
                         correct_oversampling=correct_oversampling,
                         beaming_alg=beaming_alg, save_result=save_result)
                except AttributeError:
                    pass
                
                # Quit here!
                return None
            else:
                pbdep = self.params['pbdep']['ifdep'][ref]
                        
            # Else, we have found the observations (from somewhere), and this
            # Body has spdeps attached: so we can finally compute the spectra
            base, ref = self.get_parset(ref=ref, type='syn')
            if obs['ref'] != pbdep['ref']:
                raise ValueError("IF: Something went wrong here! The obs don't match the pbdep...")
            
            # do we want to keep figures?
            keepfig = obs.get('images', '')
            
            # Retrieve the times of observations, the baseline coordinates
            # (baseline length and position angle) and effective wavelength
            times = obs['time']
            ucoord, vcoord = np.asarray(obs['ucoord']), np.asarray(obs['vcoord'])
            posangle = np.arctan2(vcoord, ucoord)/pi*180.
            baseline = sqrt(ucoord**2 + vcoord**2)
            eff_wave = None if (not 'eff_wave' in obs or not len(obs['eff_wave'])) else np.asarray(obs['eff_wave'])
            
            # If time is not given, assume all baselines are measured at
            # the same time (or equivalently the system is time independent)
            if time is None:
                keep = np.ones(len(posangle),bool)
            
            # Else, regard all time differences below 1e-8 seconds as
            # negligible
            else:
                keep = np.abs(times-time)<1e-8
                
            # If nothing needs to be computed, don't do it
            if sum(keep) == 0:
                return None
            
            # Do we need to compute closure phases?
            if 'closure_phase' in obs and len(obs['closure_phase'])==len(obs['time']):
                do_closure_phases = True
                
                # Select the observations with closure phases (first select the
                # ones with the correct timing)
                obs_ = obs[keep]
                has_closure_phase = -np.isnan(obs_['closure_phase'])
                obs_ = obs[has_closure_phase]
                
                # Get the first and second baseline and derive the third
                # baseline
                ucoord_1, vcoord_1 = np.asarray(obs_['ucoord']),\
                                     np.asarray(obs_['vcoord'])
                ucoord_2, vcoord_2 = np.asarray(obs_['ucoord_2']),\
                                     np.asarray(obs_['vcoord_2'])
                ucoord_3, vcoord_3 = -(ucoord_1 + ucoord_2),\
                                     -(vcoord_1 + vcoord_2)
                
                # Compute position angles and baselines
                posangle_1 = np.arctan2(vcoord_1, ucoord_1)/pi*180.
                posangle_2 = np.arctan2(vcoord_2, ucoord_2)/pi*180.
                posangle_3 = np.arctan2(vcoord_3, ucoord_3)/pi*180.
                baseline_1 = sqrt(ucoord_1**2 + vcoord_1**2)
                baseline_2 = sqrt(ucoord_2**2 + vcoord_2**2)
                baseline_3 = sqrt(ucoord_3**2 + vcoord_3**2)
                
                # Keep track of how many single baselines there are. For these
                # we don't need to compute closure phases afterwards
                keep = keep & np.isnan(obs['closure_phase'])
                n_single = sum(keep)
                
                # Throw all baselines (single and closed) together for
                # computations
                posangle = np.hstack([posangle[keep], posangle_1, posangle_2, posangle_3])
                baseline = np.hstack([baseline[keep], baseline_1, baseline_2, baseline_3])
                if eff_wave is not None:
                    eff_wave = np.hstack([eff_wave[keep], obs_['eff_wave'],
                                          obs_['eff_wave'], obs_['eff_wave']])
                
            else:
                # Here we only have single baselines
                do_closure_phases = False
                posangle = posangle[keep]
                baseline = baseline[keep]
                if eff_wave is not None:
                    eff_wave = eff_wave[keep]
            
            
            # make sure each time image has a unique name
            if keepfig:
                keepfig = ('{}_time_{:.8f}'.format(keepfig,time)).replace('.','_')
            else:
                keepfig = False
            
            output = observatory.ifm(self, posangle=posangle,
                                     baseline=baseline,eff_wave=eff_wave,
                                     ref=ref, keepfig=keepfig)
            
            # If nothing was computed, don't do anything
            if output is None:
                return None
            
            # Separate closure phases from single baselines
            if do_closure_phases and save_result:
                
                # First get single baseline results
                time_ = list(times[keep][:n_single])
                ucoord = list(ucoord[keep][:n_single])
                vcoord = list(vcoord[keep][:n_single])
                vis2 = list(output[3][:n_single])
                vphase = list(output[4][:n_single])
                total_flux = list(output[-1][:n_single])
                eff_wave_ = None if eff_wave is None else list(eff_wave[:n_single])
                
                # Then get the closure phases info
                vis2_1, vis2_2, vis2_3 = output[3][n_single:].reshape((3,-1))
                vphase_1, vphase_2, vphase_3 = output[4][n_single:].reshape((3,-1))
                total_flux_1, total_flux_2, total_flux_3 = output[-1][n_single:].reshape((3,-1))
                time_cp = list(times[keep][n_single:])
                if eff_wave is not None:
                    eff_wave_cp = eff_wave[n_single:].reshape((3,-1))[0]
                    
                # ... compute closure phases as the product of exponentials
                #     this will be overriden when calling __add__ at ifsyn
                closure_phase = np.angle(np.exp(1j*(vphase_1+vphase_2+vphase_3)))
                closure_ampl = np.sqrt(vis2_1*vis2_2*vis2_3)
                
            elif save_result:
                time_ = list(times[keep])
                ucoord = list(ucoord[keep])
                vcoord = list(vcoord[keep])
                vis2 = list(output[3]/output[-1]**2)
                vphase = list(output[4])
                total_flux = list(output[-1])
                eff_wave_ = None if eff_wave is None else list(eff_wave)
                            
            # Save results if necessary
            if save_result:
                base, ref = self.get_parset(type='syn', ref=ref)
                base['time'] += time_
                base['ucoord'] += ucoord
                base['vcoord'] += vcoord
                base['vis2'] += vis2
                base['vphase'] += vphase
                base['total_flux'] += total_flux
                if eff_wave is not None:
                    base['eff_wave'] += eff_wave_
                
                # Save closure phase info
                if do_closure_phases:
                    # first add nans for as many "time" we have to the single
                    # baseline entries of ucoord_2 etc
                    for col in ['ucoord_2', 'vcoord_2', 'vis2_2', 'vis2_3',\
                                'vphase_2', 'vphase_3', 'total_flux_2', 'total_flux_3',\
                                'closure_phase']:
                        base[col] += [np.nan]*len(time_)
                    
                    # Then we can add all our info on closure phases
                    base['time'] += time_cp # is already list
                    base['ucoord'] += list(ucoord_1)
                    base['vcoord'] += list(vcoord_1)
                    base['ucoord_2'] += list(ucoord_2)
                    base['vcoord_2'] += list(vcoord_2)
                    base['vis2'] += list(vis2_1/total_flux_1**2)
                    base['vis2_2'] += list(vis2_2/total_flux_2**2)
                    base['vis2_3'] += list(vis2_3/total_flux_3**2)
                    base['vphase'] += list(vphase_1)
                    base['vphase_2'] += list(vphase_2)
                    base['vphase_3'] += list(vphase_3)
                    base['total_flux'] += list(total_flux_1)
                    base['total_flux_2'] += list(total_flux_2)
                    base['total_flux_3'] += list(total_flux_3)
                    base['triple_ampl'] += list(closure_ampl)
                    base['closure_phase'] += list(closure_phase)
                    if eff_wave is not None:
                        base['eff_wave'] += list(eff_wave_cp)

    
    @decorators.parse_ref
    def rv(self,correct_oversampling=1,ref='allrvdep', time=None,
           beaming_alg='none', save_result=True):
        """
        Compute integrated radial velocity and add results to the pbdep ParameterSet.
        """
        #-- compute the projected intensities for all light curves.
        for lbl in ref:
            obs, lbl = self.get_parset(type='obs', ref=lbl)
            
            # If there are no obs here, traverse
            if obs is None:
                if hasattr(self,'bodies'):
                    for body in self.bodies:
                        body.rv(correct_oversampling=correct_oversampling,
                                ref=ref, time=time, beaming_alg=beaming_alg)
                continue
            
            base,lbl = self.get_parset(ref=lbl,type='syn')
            #proj_velo = self.projected_velocity(ref=lbl)
            proj_velo = observatory.radial_velocity(self, obs)
            
            if not save_result:
                continue
            
            base['time'].append(time)
            base['rv'].append(proj_velo)
            base['samprate'].append(correct_oversampling)
                
    @decorators.parse_ref
    def rv_nomesh(self,correct_oversampling=None,ref='allrvdep', time=None,
           save_result=True):
        """
        Dynamical RV computation.
        """
        for lbl in ref:
            obs, lbl = self.get_parset(type='obs', ref=lbl)
            
            # If there are no obs here, traverse
            if obs is None:
                if hasattr(self,'bodies'):
                    for body in self.bodies:
                        body.rv_nomesh(correct_oversampling=correct_oversampling,
                                ref=ref, time=time)
                continue
            
            orbits, comps = self.get_orbits()
            obj, vel, ptimes = keplerorbit.get_barycentric_hierarchical_orbit(time,
                                                 orbits, comps)
            # retrieve the systemic velocity
            pos = self.get_globals()
            base,lbl = self.get_parset(ref=lbl,type='syn')
            base['time'] = time
            base['rv'] = -vel[2] / kms_2_rsold + pos['vgamma']
            base['samprate'] = correct_oversampling
        
    
    
    
    @decorators.parse_ref
    def pl(self, ref='allpldep', time=None, obs=None, correct_oversampling=1,
           beaming_alg='none', save_result=True):
        """
        Compute Stokes profiles and add results to the pbdep ParameterSet.
        
        Modus operandi:
        
            1. For this body, see if there are any obs attached. If not,
               descend into the bodybag (if applicable) and repeat this step
               for every body.
            
            2. If we have a body(bag) with observations attached, see if there
               is a pbdep matching it. If not, descend into the bodybag
               (if applicable) and repeat this step.
            
            3. If we have a body(bag) with pbdep matching the previously found
               obs, compute the Stokes profile.
        
        For details on the computations, see :py:func:`observatory.stokes <phoebe.backend.observatory.stokes>`
        """
        # We need to get the observation ParameterSet so that we know all the
        # required info on **what** exactly to compute (**how** to compute it
        # is contained in the pbdep)
        if obs is None and hasattr(self,'params') and 'obs' in self.params \
                                    and 'plobs' in self.params['obs']:
            # Compute the Stokes profiles for all references
            for lbl in ref:
                
                # Get the observations if they are not given already
                if obs is None:
                    plobs, lbl = self.get_parset(type='obs', ref=lbl)
                    if lbl is None:
                        continue
                else:
                    plobs = obs
                
                # Compute the Stokes profiles for this reference
                self.pl(ref=lbl, time=time, obs=plobs)
        
        # If no obs are given and there are no obs attached, assume we're a
        # BodyBag and descend into ourselves:
        elif obs is None:
            try:
                for body in self.bodies:
                    body.pl(ref=ref, time=time, obs=obs)
            except AttributeError:
                pass
        
        # If obs are given, there is no need to look for the references, and we
        # can readily compute the Stokes profiles
        else:
            ref = obs['ref']
            
            # Well, that is, we will only compute the Stokes if there are pbdeps
            # on this body. If not, we assume it's a BodyBag and descend into
            # the bodies.
            if not (hasattr(self,'params') and 'pbdep' in self.params \
                                    and 'pldep' in self.params['pbdep']):
                try:
                    for body in self.bodies:
                        body.pl(ref=ref, time=time, obs=obs)
                except AttributeError:
                    pass
                
                # Quit here!
                return None
            else:
                pbdep = self.params['pbdep']['pldep'][ref]
                        
            # Else, we have found the observations (from somewhere), and this
            # Body has pldeps attached: so we can finally compute the Stokes
            # profiles
            base, ref = self.get_parset(ref=ref, type='syn')
            if obs['ref'] != pbdep['ref']:
                raise ValueError("PL: Something went wrong here! The obs don't match the pbdep...")
            output = observatory.stokes(self, obs, pbdep)
                
            # If nothing was computed, don't do anything
            if output is None:
                return None
            
            if save_result:
                # Expand output and save it to the synthetic thing
                wavelengths_, I, V, Q , U, cont = output
            
                base['time'].append(time)
                base['wavelength'].append(wavelengths_ / 10.)
                base['flux'].append(I)
                base['V'].append(V)
                base['Q'].append(Q)
                base['U'].append(U)
                base['continuum'].append(cont)
    
    @decorators.parse_ref
    def sp(self, ref='allspdep', time=None, obs=None, correct_oversampling=1,
           beaming_alg='none', save_result=True):
        """
        Compute spectrum and add results to the pbdep ParameterSet.
        
        Modus operandi:
        
            1. For this body, see if there are any obs attached. If not,
               descend into the bodybag (if applicable) and repeat this step
               for every body.
            
            2. If we have a body(bag) with observations attached, see if there
               is a pbdep matching it. If not, descend into the bodybag
               (if applicable) and repeat this step.
            
            3. If we have a body(bag) with pbdep matching the previously found
               obs, compute the spectrum.
        
        For details on the computations, see :py:func:`observatory.spectrum <phoebe.backend.observatory.spectrum>`
        """
        # We need to get the observation ParameterSet so that we know all the
        # required info on **what** exactly to compute (**how** to compute it
        # is contained in the pbdep)
        if obs is None and hasattr(self,'params') and 'obs' in self.params \
                                    and 'spobs' in self.params['obs']:
            # Compute the Stokes profiles for all references
            for lbl in ref:
                
                # Get the observations if they are not given already
                if obs is None:
                    spobs, lbl = self.get_parset(type='obs', ref=lbl)
                    if lbl is None:
                        continue
                else:
                    spobs = obs
                
                # Compute the Stokes profiles for this reference
                self.sp(ref=lbl, time=time, obs=spobs)
        
        # If no obs are given and there are no obs attached, assume we're a
        # BodyBag and descend into ourselves:
        elif obs is None:
            try:
                for body in self.bodies:
                    body.sp(ref=ref, time=time, obs=obs)
            except AttributeError:
                pass
        
        # If obs are given, there is no need to look for the references, and we
        # can readily compute the spectrum
        else:
            ref = obs['ref']
            
            # Well, that is, we will only compute the Stokes if there are pbdeps
            # on this body. If not, we assume it's a BodyBag and descend into
            # the bodies.
            if not (hasattr(self,'params') and 'pbdep' in self.params \
                                    and 'spdep' in self.params['pbdep']):
                try:
                    for body in self.bodies:
                        body.sp(ref=ref, time=time, obs=obs)
                except AttributeError:
                    pass
                
                # Quit here!
                return None
            else:
                pbdep = self.params['pbdep']['spdep'][ref]
                        
            # Else, we have found the observations (from somewhere), and this
            # Body has spdeps attached: so we can finally compute the spectra
            base, ref = self.get_parset(ref=ref, type='syn')
            if obs['ref'] != pbdep['ref']:
                raise ValueError("SP: Something went wrong here! The obs don't match the pbdep...")
            output = observatory.spectrum(self, obs, pbdep)
                
            # If nothing was computed, don't do anything
            if output is None:
                return None
                
            # Expand output and save it to the synthetic thing
            wavelengths_, I, cont = output
            
            if save_result:
                base['time'].append(time)
                base['wavelength'].append(wavelengths_ / 10.)
                base['flux'].append(I)
                base['continuum'].append(cont)
    
    @decorators.parse_ref
    def am(self, ref='allamdep', time=None, obs=None, correct_oversampling=1,
           beaming_alg='none', save_result=True):
        """
        Compute astrometry and add results to the pbdep ParameterSet.
        
        For details on the computations, see :py:func:`observatory.astrometry <phoebe.backend.observatory.astrometry>`
        """
        # We need to get the observation ParameterSet so that we know all the
        # required info on **what** exactly to compute (**how** to compute it
        # is contained in the pbdep)
        if obs is None and hasattr(self,'params') and 'obs' in self.params \
                                    and 'amobs' in self.params['obs']:
            # Compute the apparent position
            for lbl in ref:
                # Get the observations if they are not given already
                if obs is None:
                    amobs, lbl = self.get_parset(type='obs', ref=lbl)
                    if lbl is None:
                        continue
                else:
                    amobs = obs
                
                # Compute the Astrometry for this reference
                self.am(ref=lbl, time=time, obs=amobs)
        
        # If no obs are given and there are no obs attached, assume we're a
        # BodyBag and descend into ourselves:
        elif obs is None:
            try:
                for body in self.bodies:
                    body.am(ref=ref, time=time, obs=obs)
            except AttributeError:
                pass
        
        # If obs are given, there is no need to look for the references, and we
        # can readily compute the astrometry
        else:
            ref = obs['ref']
            
            # Well, that is, we will only compute the astrometry if there are
            # pbdeps on this body. If not, we assume it's a BodyBag and descend
            # into the bodies.
            if not (hasattr(self,'params') and 'pbdep' in self.params \
                                    and 'amdep' in self.params['pbdep']):
                try:
                    for body in self.bodies:
                        body.am(ref=ref, time=time, obs=obs)
                except AttributeError:
                    pass
                
                # Quit here!
                return None
            else:
                pbdep = self.params['pbdep']['amdep'][ref]
                        
            # Else, we have found the observations (from somewhere), and this
            # Body has amdeps attached: so we can finally compute the astrometry
            base, ref = self.get_parset(ref=ref, type='syn')
            if obs['ref'] != pbdep['ref']:
                raise ValueError("AM: Something went wrong here! The obs don't match the pbdep...")
            index = np.argmin(np.abs(obs['time']-time))
            if not obs['time'][index]==time:
                raise ValueError(("Cannot compute astrometry at time {}, not "
                                  "given in obs"))
            output = observatory.astrometry(self, obs, pbdep, index)
                
            if save_result:   
                # Save output to the synthetic thing
                base['time'].append(time)
                base['delta_ra'].append(output['delta_ra'][0])
                base['delta_dec'].append(output['delta_dec'][0])
                base['plx_lambda'].append(output['plx_lambda'][0])
                base['plx_beta'].append(output['plx_beta'][0])
    
    #}
    
    

class PhysicalBody(Body):
    """
    Extends a Body with extra functions relevant to physical bodies.
    
    Additional functionality:
    
    .. autosummary::
    
        correct_time
        add_pbdeps
        add_obs
        remove_pbdeps
        remove_obs
        remove_mesh
        reset_mesh
        update_mesh
        subdivide
        unsubdivide
        prepare_reflection
        clear_reflection
        as_point_source
       
       
    """
    def correct_time(self):
        """
        Correct time for light time travel effects.
        
        The light time travel effects are always computed with respect to
        the barycentre. If we know the distance of the object from the
        barycentre, we can compute how long the light has traveled towards
        it. Note that it is only the direction in the line-of-sight that
        matters::
        
            t_corr = distance/cc
        
        """
        #-- what's our distance from the barycentre?
        mydistance = np.average(self.mesh['center'][:,2],weights=self.mesh['size'])
        correction = mydistance/constants.cc*constants.Rsol/(24*3600.)
        #-- add the time correction so that the time is corrected for the
        #   light travel time.
        self.time += correction
        
        logger.info('light travel time at LOS distance {:.3f} Rsol from barycentre: {:+.3g} min'.format(mydistance,correction*24*60))
    
    def clear_from_reset(self, key):
        if key in self._clear_when_reset:
            self._clear_when_reset.pop(key)
    
    def get_proper_time(self, time):
        """
        Convert barycentric time to proper time for this object.
        
        The proper times need to be precomputed and stored in the ``orbsyn``
        parameterSet in the ``syn`` section of the ``params`` attribute.
        
        @param time: barycentric time
        @type time: float
        @return: proper time
        @rtype: float
        """
        if hasattr(self, 'params') and 'syn' in self.params and 'orbsyn' in self.params['syn']:
            bary_time = self.params['syn']['orbsyn'].values()[0]['bary_time']
            prop_time = self.params['syn']['orbsyn'].values()[0]['prop_time']
            # Possibly ltt where used before, and now an empty orbsyn set is
            # lying around
            if len(bary_time) == 0:
                return time
            index = np.searchsorted(bary_time, time)
            if bary_time[index] == time:
                logger.info("Barycentric time {:.10f} corrected to proper time {:.10f} ({:.6e} sec)".format(time, prop_time[index], (time-prop_time[index])*24*3600))
                return prop_time[index]
            # If not precalculated, we need to calculate it now
            else:
                out = self.get_orbits()
                objs, vels, prop_times = \
                keplerorbit.get_barycentric_hierarchical_orbit(np.array([time]),
                                out[0], out[1])
                return prop_times[0]
                #raise ValueError('Proper time corresponding to barycentric time {} not found'.format(time))
        else:
            return time
    
    
    def get_barycenter(self):
        """
        Numerically computes the center of the body from the mesh (at the current time)
        """
        
        return np.average(self.mesh['center'][:,2],weights=self.mesh['size'])
    
    def get_orbit(self):
        """
        Get the orbit for this Body.
        
        Implemented by "get_orbits"
        """
        if 'orbit' in self.params:
            return self.params['orbit']
        else:
            return None
        
    def add_pbdeps(self, pbdep, take_defaults=None):
        """
        Add a list of dependable ParameterSets to the Body.
        """
        #-- add dependables to params
        parsed_refs = _parse_pbdeps(self, pbdep, take_defaults=take_defaults)
        #-- add columns to mesh
        if len(self.mesh):
            for ref in parsed_refs:
                new_name = 'ld_{0}'.format(ref),'f8',(Nld_law,)
                
                # don't bother if it allready exists
                if new_name[0] in self.mesh.dtype.names:
                    continue
                
                dtypes = [new_name]
                dtypes.append(('proj_{0}'.format(ref),'f8'))
                #dtypes.append(('velo_{0}_'.format(ref),'f8',(3,)))
                #dtypes.append(('_o_velo_{0}_'.format(ref),'f8',(3,)))
                dtypes = np.dtype(dtypes)
                new_cols = np.zeros(len(self.mesh),dtype=dtypes)
                for i,field in enumerate(new_cols.dtype.names):
                    self.mesh = pl.mlab.rec_append_fields(self.mesh,field,new_cols[field],dtypes=dtypes[i])
        logger.info('added pbdeps {0}'.format(parsed_refs))
        return parsed_refs
    
    
    def remove_pbdeps(self,refs):
        """
        Remove dependable ParameterSets from the Body.
        """
        refs = set(refs)
        for dep in self.params['pbdep']:
            keys = set(self.params['pbdep'][dep].keys())
            intersect = list(keys & refs)
            while intersect:
                ref = intersect.pop()
                self.params['pbdep'][dep].pop(ref)
                self.params['syn'][dep[:-3]+'syn'].pop(ref)
                #-- drop fields
                fields = 'ld_{0}'.format(ref),\
                         'lproj_{0}'.format(ref),\
                         'velo_{0}'.format(ref),\
                         '_o_velo_{0}'.format(ref)
                self.mesh = pl.mlab.rec_drop_fields(self.mesh,fields)
                logger.info('removed pbdeps {0}'.format(fields))
    
    def add_mesh_fields(self, fields):
        dtypes = np.dtype(self.mesh.dtype.descr + fields)
        onames = self.mesh.dtype.names
        omesh = self.mesh.copy()
        N = len(omesh)
        self.mesh = np.zeros(N,dtype=dtypes)
        if N>0:
            self.mesh[onames] = omesh[onames]
        
    
    def remove_mesh(self):
        self.mesh = np.zeros(0,dtype=self.mesh.dtype)
    
    
    @decorators.parse_ref
    def prepare_reflection(self, ref=None):
        """
        Prepare the mesh to handle reflection from an other body.
        
        We only need one extra column with the incoming flux divided by pi
        to account for isotropic scattering. Doppler beaming and such should
        be taken into account in the reflection algorithm. In the isotropic case,
        reflections is aspect indepedent.
        
        """
        for iref in ref:
            field = 'refl_{}'.format(iref)
            if field in self.mesh.dtype.names:
                continue
            dtypes = np.dtype([(field, 'f8')])
            new_cols = np.zeros(len(self.mesh),dtype=np.dtype(dtypes))
            self.mesh = pl.mlab.rec_append_fields(self.mesh,field,new_cols[field])
            logger.info('added reflection column {} for pbdep {} to {}'.format(field, iref, self.get_label()))
    
    
    @decorators.parse_ref
    def clear_reflection(self,ref='all'):
        """
        Reset the reflection columns to zero.
        """
        for iref in ref:
            field = 'refl_{}'.format(iref)
            if field in self.mesh.dtype.names:
                self.mesh[field] = 0.0
                logger.info('Emptied reflection column {}'.format(iref))
                    
    @decorators.parse_ref
    def prepare_beaming(self,ref=None):
        """
        Prepare the mesh to handle beaming.
        """
        for iref in ref:
            field = 'alpha_b_{}'.format(iref)
            if field in self.mesh.dtype.names:
                continue
            dtypes = np.dtype([(field,'f8')])
            new_cols = np.zeros(len(self.mesh),dtype=np.dtype(dtypes))
            self.mesh = pl.mlab.rec_append_fields(self.mesh,field,new_cols[field])
            logger.debug('added beamin column for pbdep {}'.format(iref))
    
    
    
    def as_point_source(self,only_coords=False,ref=0):
        """
        Return a point-source representation of the PhysicalBody.
        
        This is a generic function that should be valid for some basic Bodies;
        for custom defined Bodies, this function should probably be redefined.
        You'll know when its needed if this function gives an error :-).
        
        The following things are (or should/could be) computed:
            1.  geometric barycentre in 3D space: these are the average
                coordinates of all triangle centers, weighed with the triangle
                sizes
            2.  total passband luminosity
            3.  Projected passband flux
            4.  on-sky photocenter: average coordinates of the triangle centers
                weighted with the projected intensity (only (x,y))
            5.  mean observed passband temperature: average temperature weighed
                with the projected intensity
            6.  mean observed surface gravity: average surface gravity weighed
                with the projected intensity
            7.  mass (Msol)
            8.  distance (pc)
            9.  proper motion of photocenter (mas/yr) 
            10. velocity of geometric barycenter in 3D space (km/s)
        
        We should replace the output here by a ParameterSet, that's what they
        are for!
        
        (x,y,z), (vx,vy,vz), mass (Msol), luminosity (Lsol)
        """
        if only_coords:
            return np.average(self.mesh['center'],weights=self.mesh['size'],axis=0)
        #-- get the information on the passband, and compute the weights that
        #   are used for the projected intensity
        idep,ref = self.get_parset(ref=ref)
        logger.info('Representing Body as point source in band {}'.format(ref))
        proj_int = self.projected_intensity(ref=ref)
        wflux = self.mesh['proj_'+ref]
        wsize = self.mesh['size']
        
        ps = parameters.ParameterSet(context='point_source')
        #-- distance of axes origin
        myglobals = self.get_globals()
        if myglobals is not None:
            deltaz = myglobals.request_value('distance', 'Rsol')
        else:
            deltaz = 0.
        origin = np.array([0,0,deltaz])
        
        #-- 1-3.  Geometric barycentre and photocentric barycentre
        ps['coordinates'] = np.average(self.mesh['center'],weights=wsize,axis=0)+origin,'Rsol'
        ps['photocenter'] = np.average(self.mesh['center'],weights=wflux*wsize,axis=0)+origin,'Rsol'
        ps['velocity']    = np.average(self.mesh['velo___bol_'],axis=0),'Rsol/d'
        ps['distance']    = sqrt(ps['coordinates'][0]**2+ps['coordinates'][1]**2+ps['coordinates'][2]**2),'Rsol'
        #-- 4.  mass: try to call the function specifically designed to compute
        #       the mass, if the class implements it.
        try:
            ps['mass'] = self.get_mass(),'Msol'
        except AttributeError:
            if 'mass' in self.params.values()[0]:
                ps['mass'] = self.params.values()[0].get_value_with_unit('mass')
            else:
                ps['mass'] = 0.
        #-- 5.  mean observed passband temperature
        ps['teff'] = np.average(self.mesh['teff'],weights=wflux,axis=0),'K'
        #-- 6.  mean observed passband surface gravity
        ps['surfgrav'] = np.average(self.mesh['logg'],weights=wflux,axis=0)
        #-- 7.  projected flux
        ps['intensity'] = proj_int
        
        logger.info('... mass = {mass} Msol'.format(**ps))
        logger.info('... photocenter = {photocenter}'.format(**ps))
        logger.info('... passband temperature = {teff}'.format(**ps))
        logger.info('... passband surface gravity = {surfgrav}'.format(**ps))
        
        #if 'body' in self.params and 't_pole' in self.params['body']:
            #teff = self.params['body'].request_value('t_pole','K')
            #grav = self.params['body'].request_value('g_pole','m/s2')
        #elif 'body' in self.params and 'teff' in self.params['body']:
            #teff = self.params['body'].request_value('teff','K')
            #grav = 100.#self.params['body'].request_value('surfgrav','m/s2')
        
        #-- average radius
        barycentre = (ps.get_value('coordinates','Rsol')-origin).T 
        ps['radius'] = coordinates.norm(self.mesh['center']-barycentre).mean(),'Rsol'
        
        
        return ps
    
    def get_passband_gravity_brightening(self, ref=0, blackbody=False):
        r"""
        Compute the passband gravity brightening.
        
        It is defined as in e.g. [Claret2011]_:
        
        .. math::
        
            y = \left(\frac{\partial\ln I(\lambda)}{\partial\ln g}\right)_{T_\mathrm{eff}} + \left(\frac{d\ln T_\mathrm{eff}}{d\ln g}\right)\left(\frac{\partial\ln I(\lambda)}{\partial\ln T_\mathrm{eff}}\right)_g
        
        However, in our numerical model, we don't have the partial derivatives,
        so we compute it as:
        
        .. math::
        
            y = \frac{\Delta \ln I}{\sqrt{(\Delta\ln T_\mathrm{eff})^2 + (\Delta\ln g)^2}}
            
        """
        parset, ref = self.get_parset(ref=ref)
        dlnI = np.log(self.mesh['ld_'+ref][:,-1]).ptp()
        dlnT = np.log(self.mesh['teff']).ptp()
        dlng = np.log(10**self.mesh['logg']).ptp()
        if blackbody:
            dlng = 0
            
        if dlnT == 0 and dlng == 0:
            passband_gravb = 0.0
        else:
            passband_gravb = dlnI / np.sqrt(dlnT**2 + dlng**2)
        
        #print("passband gravity darkening, {}+{}*{}={}".format(dlnI_dlng,dlnT_dlng,dlnI_dlnT, passband_gravb))
        return passband_gravb
    
    
    def init_mesh(self):
        init_mesh(self)
    
    def reset_mesh(self):
        """
        Reset the mesh to its original position.
        """
        columns = self.mesh.dtype.names

        # All column names starting with _o_ are mapped to the ones without
        # _o_.
        self.mesh['center'] = self.mesh['_o_center']
        self.mesh['triangle'] = self.mesh['_o_triangle']
        self.mesh['normal_'] = self.mesh['_o_normal_']
        self.mesh['velo___bol_'] = self.mesh['_o_velo___bol_']
        self.mesh['size'] = self.mesh['_o_size']
        
        if 'B_' in columns:
            self.mesh['B_'] = self.mesh['_o_B_']
        
        #for column in columns:
        #   if column[:3] == '_o_' and column[3:] in columns:
        #       self.mesh[column[3:]] = self.mesh[column]
        
        # And we know nothing about the visibility
        self.mesh['partial'] = False
        self.mesh['visible'] = False
        self.mesh['hidden'] = True
    
    
    def update_mesh(self,subset=None):
        """
        Update the mesh for a subset of triangles or the whole mesh
        """
        if subset is None:
            subset = np.ones(len(self.mesh),bool)
        #-- cut out the part that needs to be updated
        old_mesh = self.mesh[subset]#.copy()
        #-- remember which arguments were used to create the original mesh
        mesh_args = self.subdivision['mesh_args']
        mesh_args,scale = mesh_args[:-1],mesh_args[-1]
        #logger.info('updating %d/%d triangles in mesh with args %s (scale=%s)'%(sum(subset),len(self.mesh),str(mesh_args),str(scale)))
        #logger.info('updating some triangles in mesh with args %s (scale=%s)'%(str(mesh_args),str(scale)))
        #-- then reproject the old coordinates. We assume they are fairly
        #   close to the real values.
        
        #-- C or Python implementation:
        if True:
            # Size doesn't matter -- I mean it's not taken into account during
            # reprojection
            select = ['_o_center','_o_size','_o_triangle','_o_normal_']
            old_mesh_table = np.column_stack([old_mesh[x] for x in select])/scale
            old_mesh_table = marching.creproject(old_mesh_table,*mesh_args)*scale
            
            #old_mesh_table = np.column_stack([old_mesh[x] for x in select])
            #old_mesh_table = marching.creproject_extra(old_mesh_table[select], scale,*mesh_args)
            # Check direction of normal
            #cosangle = coordinates.cos_angle(old_mesh['_o_center'],
            #                                 old_mesh_table[:,13:16],axis=1)
            #sign = np.where(cosangle<0,-1,1).reshape((-1,1))
            
            
            for prefix in ['_o_','']:
                old_mesh[prefix+'center'] = old_mesh_table[:,0:3]
                old_mesh[prefix+'triangle'] = old_mesh_table[:,4:13]
                old_mesh[prefix+'normal_'] = -old_mesh_table[:,13:16]
            
            
        #-- Pure Python (old): I keep it because there might be issues with the
        #   direction of the normals that I haven't checked yet.
        else:
        
            for tri in range(len(old_mesh)):
                p0 = marching.projectOntoPotential(old_mesh['_o_center'][tri]/scale,*mesh_args)
                t1 = marching.projectOntoPotential(old_mesh['_o_triangle'][tri][0:3]/scale,*mesh_args)
                t2 = marching.projectOntoPotential(old_mesh['_o_triangle'][tri][3:6]/scale,*mesh_args)
                t3 = marching.projectOntoPotential(old_mesh['_o_triangle'][tri][6:9]/scale,*mesh_args)
                #-- make sure the normal is pointed in the same direction as before:
                cosangle = coordinates.cos_angle(old_mesh['_o_center'][tri:tri+1],
                                                np.array([p0.n]),axis=1)
                #cosangle = cgeometry.cos_theta(old_mesh['_o_center'][tri:tri+1].ravel(order='F').reshape((-1,3)),
                #                               np.array([p0.n]))
                sign = cosangle<0 and -1 or 1
                for prefix in ['_o_','']:
                    old_mesh[prefix+'center'][tri] = p0.r*scale
                    old_mesh[prefix+'normal_'][tri] = sign*p0.n
                    old_mesh[prefix+'triangle'][tri][0:3] = t1.r*scale
                    old_mesh[prefix+'triangle'][tri][3:6] = t2.r*scale
                    old_mesh[prefix+'triangle'][tri][6:9] = t3.r*scale
                
        #-- insert the updated values in the original mesh
        #mlab.figure()
        #self.plot3D(normals=True)    
        
        self.mesh[subset] = old_mesh
        
        # Normals and centers are updated, but sizes are not.
        self.compute_sizes(prefix='_o_')
        self.mesh['size'] = self.mesh['_o_size']
        #self.compute_sizes(prefix='')
        
        
    
    def subdivide(self,subtype=0,threshold=1e-6,algorithm='edge'):
        """
        Subdivide the partially visible triangles.
        
        We need to keep track of the original partially visible triangles so
        that at any point, we can revert to the original, unsubdivided mesh.
        
        Subtype is an integer, meaning:
            
            - C{subtype=0}: only subdivide the current mesh
            - C{subtype=1}: only subdivide the original mesh
            - C{subtype=2}: subdivide both current and original mesh
        
        Complexity is a number, meaning:
        
            - complexity=0: roughest subdivision, just subdivide and let the
              subdivided triangles keep all properties of the parent triangles,
              except their size
        
            - complexity=1: subdivide but reproject the centers and vertices,
              and recompute normals.
        
            - complexity=2: subdivide and recalculate all quantities of the
              subdivided triangles
        
        Algorithm is a string, but should be set to C{edge} unless you know
        what you're doing.
        """
        if subtype==0:
            prefix = ''
        elif subtype==1:
            prefix = '_o_'
        elif subtype==2:
            threshold = 0
            prefix = ['_o_','']
        logger.debug("subdividing: type {:d} via {:s}".format(subtype,algorithm))
        #-- subidivde the partially visible triangles
        partial = self.mesh['partial']
        subdivided = subdivision.subdivide(self.mesh[partial],prefix=prefix,
              threshold=threshold,algorithm=algorithm)
        #-- orientate the new triangles in the universe (rotation + translation)
        if len(subdivided):
            #-- replace old triangles with newly subdivided ones, but remember the
            #   old ones if this is the first time we subdivide the mesh
            if self.subdivision['orig'] is None:
                self.subdivision['orig'] = self.mesh#.copy()
            self.mesh = np.hstack([self.mesh[-partial],subdivided])
            if subtype==1:
                #self.update_mesh(self.mesh['partial'])
                self.rotate_and_translate(subset=self.mesh['partial'])
                logger.debug('rotated subdivided mesh')
        return len(subdivided)
    
    
    def unsubdivide(self):
        """
        Revert to the original, unsubdivided mesh.
        """
        if self.subdivision['orig'] is None:
            logger.debug('nothing to unsubdivide')
        else:
            self.mesh = self.subdivision['orig']
            logger.debug("restored mesh from subdivision")
        self.subdivision['orig'] = None
    
    def add_systemic_velocity(self, grav=False):
        """
        Add the systemic velocity component and gravitational redshift to the
        system.
        
        Notice that vgamma is in the opposite direction of our definition!
        """
        # Add systemic velocity:
        globs = self.get_globals()
        
        if globs is not None:
            #vgamma = globals.request_value('vgamma', 'Rsol/d')
            vgamma = globs['vgamma'] * kms_2_rsold
            if vgamma != 0:
                self.mesh['velo___bol_'][:,2] -= vgamma
        
        # Gravitational redshift
        if grav:
            radius = coordinates.norm(self.mesh['center'], axis=1)*constants.Rsol
            rv_grav = constants.GG*self.get_mass()*constants.Msol/radius/constants.cc / constants.Rsol*24*3600.
            self.mesh['velo___bol_'][:,2] += rv_grav
        
        
    
    def remove_systemic_velocity(self, grav=False):
        """
        Remove the systemic velocity component and gravitational redshift from
        the system.
        """
        # Remove systemic velocity:
        globals = self.get_globals()
        if globals is not None:
            #vgamma = globals.request_value('vgamma', 'Rsol/d')
            vgamma = globals['vgamma'] * 1000. / constants.Rsol * 24 * 3600
            self.mesh['velo___bol_'][:,2] += vgamma
        # Gravitational redshift
        if grav:
            radius = coordinates.norm(self.mesh['center'], axis=1)*constants.Rsol
            rv_grav = constants.GG*self.get_mass()*constants.Msol/radius/constants.cc / constants.Rsol*24*3600.
            self.mesh['velo___bol_'][:,2] += rv_grav
    
    def save_syn(self,filename,category='lc',ref=0,sigma=None,mode='w'):
        """
        Save synthetic data.
        """
        ds,ref = self.get_parset(category=category,type='syn',ref=ref)
        pb,ref = self.get_parset(category=category,type='pbdep',ref=ref)
        
        #-- add errors if needed
        if sigma is not None:
            if 'sigma' not in ds['columns']:
                ds['columns'].append('sigma')
            ds['sigma'] = sigma
        
        #-- open the filename or the stream
        if isinstance(filename,str):
            ff = open(filename,mode)
        else:
            ff = filename
        
        #-- write the parameters
        for key in pb:
            par = pb.get_parameter(key)
            ff.write('# {:s} = {:s}\n'.format(par.get_qualifier(),par.to_str()))
            print('# {:s} = {:s}\n'.format(par.get_qualifier(),par.to_str()))
        
        #-- write the dataset
        ds.save(ff,pretty_header=True)
        
        #-- clean up
        if isinstance(filename,str):
            ff.close()
    
    #{ Functions to compute dependables
    def get_parameters(self,ref=0):
        """
        Return a copy of the Body parameters, plus some extra information:
        
            - mean projected temperature
            - mean projected gravity
            - stellar luminosity
        """
        body = list(self.params.values())[0]
        params = body.copy()
        ps,ref = self.get_parset(ref)
        #-- luminosity: we first compute the luminosity as if it were
        #   constant along the surface
        Rsol_cm = constants.Rsol*100.
        t_pole = body.request_value('t_pole','K')
        r_pole = body.request_value('r_pole','m')
        #keep = self.mesh['mu']>0
        #lumi = 4*pi*(self.mesh['ld'][:,-1]*self.mesh['size']*constants.Rsol_cgs**2).sum()/constants.Lsol_cgs
        #print self.mesh['size'].sum()
        #print lumi
        #raise SystemExit
        ##   due to local changes, the luminosity can get biased. See
        #   e.g. Eq 30 and 31 of Cranmer and Owocki, 1995
        g_pole = body.request_value('g_pole')
        g_avrg = np.average(10**(self.mesh['logg']-2),weights=self.mesh['size'])
        lumi = (4*pi*r_pole**2*constants.sigma*t_pole**4)/constants.Lsol
        lumi = g_avrg/g_pole*lumi
        params.add(parameters.Parameter(qualifier='luminosity',value=lumi,unit='Lsol',description='Total luminosity'))
        #-- mean temperature/gravity
        #Tmean = np.average(self.mesh['teff'],weights=self.mesh['ld'][:,-1]*self.mesh['size'])
        #Gmean = np.average(self.mesh['logg'],weights=self.mesh['ld'][:,-1]*self.mesh['size'])
        #print Tmean,Gmean
        Tmean = np.average(self.mesh['teff'],weights=self.mesh['proj_'+ref]*self.mesh['size'])
        Gmean = np.average(self.mesh['logg'],weights=self.mesh['proj_'+ref]*self.mesh['size'])
        params.add(parameters.Parameter(qualifier='teffobs',value=Tmean,unit='K',description='Flux-weighted integrated effective temperature'))
        params.add(parameters.Parameter(qualifier='loggobs',value=Gmean,unit='',description='Flux-weighted integrated surface gravity'))
        
        return params    

    
    

    
    def get_obs(self,category=None,ref=0):
        """
        Retrieve data.
        """
        #if pbdeptype.endswith('dep'):
        #    pbdeptype = pbdeptype[:-3]+'obs'
        base,ref = self.get_parset(ref=ref,type='obs',category=category)
        return base
    
        
    @decorators.parse_ref
    def lc(self, correct_oversampling=1, ref='alllcdep', time=None,
           beaming_alg='none', save_result=True):
        """
        Compute projected intensity and add results to the pbdep ParameterSet.
        
        """
        #-- don't bother if we cannot do anything...
        if hasattr(self,'params') and 'pbdep' in self.params:
            if not ('lcdep' in self.params['pbdep']):
                return None
            #-- compute the projected intensities for all light curves.
            for lbl in ref:
                base,lbl = self.get_parset(ref=lbl,type='syn')
                proj_intens = self.projected_intensity(ref=lbl, beaming_alg=beaming_alg)
                
                if not save_result:
                    continue
                
                base['time'].append(time)
                base['flux'].append(proj_intens)
                base['samprate'].append(correct_oversampling)
    

    
    @decorators.parse_ref
    def ps(self,correct_oversampling=1, ref='alllcdep',time=None, beaming_alg='none'):
        """
        Compute point-source representation of Body.
        """
        if hasattr(self,'params') and 'pbdep' in self.params:
            #-- compute the projected intensities for all light curves.
            for lbl in ref:
                myps = self.as_point_source(ref=lbl)
                #base,lbl = self.get_parset(ref=lbl,type='syn')
                for qualifier in myps:
                    self.params['pbdep']['psdep'][lbl]['syn']['time'].append(time)
                    
    
    
class BodyBag(Body):
    """
    Body representing a group of bodies.
    
    **Adding and removing parameters, data and bodies**
    
    .. autosummary::
    
       add_obs
       append
    
    **Resetting/clearing**
    
    .. autosummary::
        
       Body.clear_synthetic
       reset
       remove_mesh
       
    **Requesting (additional) information**
    
    .. autosummary::
    
       Body.get_parset
       Body.get_refs
       Body.get_synthetic
       Body.get_coords
       Body.list
       get_label
       get_logp
       get_component
       get_model
       get_obs
       get_lc
       get_bodies
       
    
    **Iterators**
    
    .. autosummary::
    
       Body.walk
       Body.walk_all
       walk_type
       get_bodies
    
    **Computing passband dependent quantities**
    
    .. autosummary::
    
        ifm
        
    **Body computations**
     
    .. autosummary::
       
       Body.compute_centers
       Body.compute_normals
       Body.compute_sizes
       Body.compute_scale_or_offset
       Body.detect_eclipse_horizon
    
    **Input/output**
    
    .. autosummary::
    
        Body.save
        to_string
    
    **Basic plotting** (see plotting module for more options)
    
    .. autosummary::
        
       Body.plot2D
       Body.plot3D
    
    **Section 1: Introduction**
    
    A L{BodyBag} keeps a list of all Bodies in the bag separately, but basic
    Body operations such as rotation and translation are performed on a merged
    mesh of all Bodies. Calling a function that is not defined in L{Body}
    will result in a looped call over all Bodies in the L{BodyBag}. For example,
    calling L{intensity} on a BodyBag will set the intensity of each L{Body} in
    the L{BodyBag}. This behaviour makes it easy to nest Bodies, e.g. for putting
    a system of bodies in binary orbit around another object.
    
    In practical terms, a BodyBag may be thought of as a list object with
    extra capabilities. You can index a BodyBag using integer indexing, array
    indexing and slicing. You can ask for the length
    of a BodyBag (C{len}), you can append other Bodies to it (C{append}) and
    add two BodyBags together to create a new (nested) BodyBag.
    
    **Section 2: Example usage**
    
    Let us create a BodyBag consisting of three stars. For clarity, we first
    make a list of the stars:
    
    >>> bodies = []
    >>> for i in range(3):
    ...    star_pars = parameters.ParameterSet(context='star',label='star{}'.format(i))
    ...    mesh_pars = parameters.ParameterSet(context='mesh:marching')
    ...    lcdp_pars = parameters.ParameterSet(context='lcdep')
    ...    star = Star(star_pars,mesh=mesh_pars,pbdep=[lcdp_pars])
    ...    bodies.append(star)

    A BodyBag is easily created from a list of Bodies:
    
    >>> bb = BodyBag(bodies)

    A BodyBag behaves much like a list:
    
    >>> print(len(bb))
    3
    >>> print(bb[0].params['star']['label'])
    star0
    >>> print(bb[-1].params['star']['label'])
    star2

    We can iterate over the list of bodies in a BodyBag:
    
    >>> for star in bb.bodies:
    ...    print(star.params['star']['label'])
    star0
    star1
    star2

    We can slice it:
    
    >>> bb2 = bb[::2]
    >>> for star in bb2.bodies:
    ...    print(star.params['star']['label'])
    star0
    star2
    
    Append new bodies to an existing BodyBag:
    
    >>> bb2.append(bb[1])
    >>> for star in bb2.bodies:
    ...    print(star.params['star']['label'])
    star0
    star2
    star1
    
    A BodyBag can also be created by summing Bodies. Since BodyBags are Bodies
    too, they can also be summed.
    
    >>> bb3 = bodies[1] + bodies[2]
    >>> for star in bb3.bodies:
    ...    print star.params['star']['label']
    star1
    star2
    
    B{Warning:} BodyBag summation is not associative!
    
    >>> bb4 = bodies[1] + bodies[2] + bodies[0]
    
    Will create a BodyBag consisting a BodyBag (holding C{bodies[1]} and
    C{bodies[2]}) and Body C{bodies[0]}. It is B{not} a BodyBag consisting
    of three bodies!
    """
    def __init__(self,list_of_bodies, obs=None, connected=True,
                 report_problems=False,  solve_problems=False, **kwargs):
        """
        Initialise a BodyBag.
        
        Extra keyword arguments can be Parameters that go in the C{self.params}
        dictionary of the BodyBag.
        
        @param list_of_bodies: list of bodies
        @type list_of_bodies: list
        """
        # We definitely need signals and a label, even if it's empty
        self.signals = {}
        self.label = str(uuid.uuid4())
        self.parent = None
        
        # Do the components belong together? This is important for the eclipse
        # detection algorithm. If they don't belong together, we don't need
        # to detect eclipses on the total BodyBag, but we can delegate it to the
        # components
        self.connected = connected
        
        # Make sure the list of bodies is a list
        if not isinstance(list_of_bodies, list):
            list_of_bodies = [list_of_bodies]
        self.bodies = list_of_bodies
        
        # Put a link to this Parent in each of the children:
        for body in self.bodies:
            body.set_parent(self)
        
        # The components need to be Bodies. I'm storing the "dimension" variable
        # here, but that's a left over from earlier days. It doesn't hurt to
        # have it.
        try:
            self.dim = self.bodies[0].dim
        except AttributeError:
            raise AttributeError("Components in a BodyBag need to be of type 'Body', not {}".format(type(self.bodies[0])))
        
        # Keep track of the current orientation, the original (unsubdivided)
        # mesh and all the parameters of the object.
        self.orientation = dict(theta=0, incl=0, Omega=0, pivot=(0, 0, 0),
                           los=[0, 0, +1], conv='YXZ', vector=[0, 0, 0])
        self.subdivision = dict(orig=None)
        self.params = OrderedDict()
        
        # Add globals parameters, but only if given. DO NOT add default ones,
        # that can be confusing
        if 'position' in kwargs and kwargs['position'] is not None:
            self.params['position'] = kwargs.pop('position')
        
        # Also the _plot is a leftover from earlier days, this is deprecated
        self._plot = self.bodies[0]._plot
        
        # Process any extra keyword arguments: they can contain a label for the
        # bodybag, or extra ParameterSets. We need to set a label for a BodyBag
        # if it's a binary, so that we can figure out which component it is
        # We also allow for "compute" parameters here!
        for key in kwargs:
            if kwargs[key] is None:
                continue
            if key == 'label':
                self.label = kwargs[key]
                continue
            self.params[key] = kwargs[key]
            
        # Perform a consistency check for the labels in the mesh. First we cycle
        # over all Bodies. That means traverse into BodyBags, so we can handle
        # hierarchical systems! The check is not performed because it is
        # actually not strictly necessary for the labels to be the same, but the
        # use of the class is limited. Still, one might want to collect
        # different bodies in a BodyBag only to set the time of all of them,
        # without the need for plotting or making dependables for all of them
        # together... perhaps we need a different BodyBag for this, or we
        # introduce some flags like "consistency_check=True" or so
        if report_problems:
            present_refs = []
            check_for_this_body = []
            
            # Iterator of bodies, lists, tuples and dictionaries
            iterator = utils.traverse(self.bodies,
                                      list_types=(list, tuple, BodyBag),
                                      dict_types=(dict,))
            for i,body in enumerate(iterator):
                
                # Iterate over passband depedent ParameterSets
                for ps in utils.traverse(body.params['pbdep']):
                    
                    # There cannot be two parametersets with the same ref
                    check_for_this_body.append(ps['ref'])
                    if i == 0 and ps['ref'] in present_refs:
                        raise ValueError("at least 2 dependables have reference {} in Body {}: make sure the labels are unique".format(ps['ref'],body.label))
                    elif i==0:
                        present_refs.append(ps['ref'])
                
                # Check which references are unique to this body and the other
                # bodies
                left_over = set(present_refs) - set(check_for_this_body)
                right_over = set(check_for_this_body) - set(present_refs)
                if len(left_over):
                    raise ValueError("missing dependables ({}) and/or extra dependables ({}) in Body {}".format(", ".join(left_over),", ".join(right_over),body.get_label()))        
                check_for_this_body = []
        
        # Fix the mesh if need
        if solve_problems:
            self.fix_mesh()
            
        # Prepare to hold synthetic results and/or observations.
        self.params['syn'] = OrderedDict()
        self.params['obs'] = OrderedDict()
        
        # If observations are give, parse them now
        if obs is not None:
            _parse_obs(self, obs)
        
        
        # The following list of functions will be executed before and after a
        # call to set_time
        self._preprocessing = []
        self._postprocessing = []
        
        # Add a dict that we can use to store temporary information
        self._clear_when_reset = dict()  
        self._main_period = dict()
    
    def set_connected(self, connected=True):
        self.connected = connected
    
    def __getattr__(self, name):
        """
        Pass on nonexisting functions to the individual bodies.
        
        When calling a function C{name} of BodyBag, C{__getattribute__} will be
        called first, and if the function C{name} exists, that one will be
        called. If it doesn't exist, C{__getattr__} will be called.
        C{__getattr__} is redefined here and passes on the call to the
        individual bodies, but only when the function is not private (that is,
        it doesn't begin and end with a double underscore). The latter is needed
        for unpickling; when a pickled object is loaded, the function
        C{__getnewargs__} is called, but we haven't defined it. Therefore, it
        will look for that function at a lower level, and apparantly a maximum
        recursion depth can be reached. Now, the loading sees no function
        C{__getnewargs__} and will thus not try to load it.
        """
        
        # Make sure to pass on calls to builtin functions immediately to the
        # bodybag.
        if name.startswith('__') and name.endswith('__'):
            return super(BodyBag, self).__getattr__(name)
        
        # All other functions needs to pass by CallInstruct to see if they can
        # can be called from BodyBag or from each object individually.
        else:
            return CallInstruct(name, self.bodies)
    
    
    def __iter__(self):
        """
        Makes the class iterable.
        """
        for param in list(self.params.values()):
            yield param
        for body in self.bodies:
            yield body
    
    def __iadd__(self, other):
        self.bodies.append(other)
        return self
    
    
    def walk_bodies(self):
        """
        Walk over all Bodies and BodyBags.
        """
        children = self.get_children()
        self_and_children = [self] + children
        for body in self_and_children:
            if body is self:
                yield body
            elif hasattr(body, 'get_children'):
                for sub_body in body.walk_bodies():
                    yield sub_body
            else:
                yield body
            
    
    def walk_type(self, type='syn'):
        """
        Walk through parameterSets of a certain type.
        
        This can be handy to walk through all 'syn', 'pbdep' or 'obs'.
        
        Overrides walk_type from Body
        
        @param type: type of dependable to walk through
        @type type: str, one of 'syn', 'pbdep', 'obs'
        @return: generator of ParameterSets
        @rtype: generator
        """
        if type in self.params:
            for param in list(self.params[type].values()):
                for key in param.values():
                    yield key
        for body in self.bodies:
            for x in body.walk_type(type=type):
                yield x
    
    
    def __len__(self):
        """
        The length of a BodyBag is the number of bodies it contains.
        
        @return: length of the Bodybag
        @rtype: int
        """
        return len(self.bodies)
    
    
    def __getitem__(self, key):
        """
        Implements various ways to get individual bodies.
        
        Allows integer indexing, slicing, indexing with integer and boolean
        arrays.
        
        When accessing individual Bodies (i.e. via integers), you get a Body
        back. When accessing via slicing or arrays, a BodyBag with the subset
        of bodies is returned.
        
        Raises IndexError upon invalid input.
        
        @param key: index
        @type key: int, slice, ndarray
        @return: individual Body or BodyBag of subset
        @rtype: Body or BodyBag
        """
        
        # Via slicing
        if isinstance(key, slice):
            return BodyBag([self[ii] for ii in range(*key.indices(len(self)))])
        
        # Via an integer
        elif isinstance(key, int):
            return self.bodies[key]
        
        # Else it's an array (could be a string still, we're not checking that)
        else:
            
            # Try to make the input an array
            try:
                key = np.array(key)
            except:
                raise IndexError(("Cannot use instance of type {} for "
                                  "indexing").format(type(key)))
            
            # Iinteger array slicing
            if key.dtype == np.dtype(int):
                return BodyBag([self[ii] for ii in key])
            
            # Boolean array slicing
            elif key.dtype == np.dtype(bool):
                return BodyBag([self[ii] for ii in range(len(key)) if key[ii]])
            
            # That's all I can come up with
            else:
                raise IndexError(("Cannot use arrays of type {} for "
                                  "indexing").format(key.dtype))
    
    def __str__(self):
        """
        String representation.
        """
        return self.to_string()
    
    def preprocess(self, time=None, **kwargs):
        """
        Run the preprocessors.
        
        @param time: time to which the Body will be set
        @type time: float or None
        """
        # First preprocess lower level stuff
        for body in self.bodies:
            body.preprocess(time=time)
            
        # Then this level stuff
        for func, arg, kwargs in self._preprocessing:
            getattr(processing, func)(self, time, *arg, **kwargs)
    
    
    def postprocess(self, time=None):
        """
        Run the postprocessors.
        """
        # First postprocess lower level stuff
        for body in self.bodies:
            body.postprocess(time=time)
        
        # Then this level stuff
        for func, args, kwargs in self._postprocessing:
            getattr(processing, func)(self, time, *args, **kwargs)
    
    def get_adjustable_parameters(self, with_priors=True):
        """
        Return a list of all adjustable parameters.
        """
        # First get the adjustable parameters in the BodyBag's .params attribute
        mylist = super(BodyBag, self).get_adjustable_parameters(with_priors=with_priors)
        
        # Then the adjustable from all subbodies.
        for body in self.bodies:
            this_adjustable = body.get_adjustable_parameters(with_priors=with_priors)
            
            # Make sure not to store duplicates
            body_list = [par for par in this_adjustable if not par in mylist]
            mylist += body_list
            
        return mylist
        
    
    def fix_mesh(self):
        """
        Make sure all bodies in a list have the same mesh columns.
        """
        # Make sure the mesh is initialised
        self.init_mesh()
        
        # here, we check which columns are missing from each Body's mesh. If
        # they are missing, we simply add them and copy the contents from the
        # original mesh.
        logger.debug("Preparing mesh")
        
        # Get a flattened list of all the bodies
        bodies = self.get_bodies()
        
        # Retrieve all the column names and column types from the individual
        # meshes. Start with the first Body
        names = list(bodies[0].mesh.dtype.names)
        descrs = bodies[0].mesh.dtype.descr
        
        # and then append the rest
        for b in bodies[1:]:
            descrs_ = b.mesh.dtype.descr
            for descr in descrs_:
                if descr[0] in names:
                    continue
                descrs.append(descr)                    
                names.append(descr[0])
        
        # For each Body, now reinitialize the mesh with a new mesh containing
        # all the columns from all the components
        dtypes = np.dtype(descrs)
        for b in bodies:
            N = len(b.mesh)
            new_mesh = np.zeros(N,dtype=dtypes)
            if N:
                cols_to_copy = list(b.mesh.dtype.names)
                for col in cols_to_copy:
                    new_mesh[col] = b.mesh[col]
                #new_mesh[cols_to_copy] = b.mesh[cols_to_copy]
            b.mesh = new_mesh
            
        # We need to make sure to reset the body, otherwise we could be fooled
        # into thinking that everything is still calculated! Some bodies do not
        # recalculate anything when the time is already set (because they are
        # time independent). This function effectively puts all values in the
        # columns to zero!
        self.reset()
    
    
    def remove_mesh(self):
        for body in self.bodies:
            body.remove_mesh()
        
        
    def get_bodies(self):
        """
        Return all possibly hierarchically stored bodies in a flatted list.
        """
        mylist = []
        for body in self.bodies:
            if hasattr(body,'bodies'):
                mylist += body.get_bodies()
            else:
                mylist.append(body)
        return mylist
    
    def get_children(self):
        """
        Return the immediate children of this object.
        """
        return self.bodies
    
    def get_period(self):
        """
        Retrieve the period and ephemeris of the system.
        """
        # If the user didn't set anything we can make smart defaults
        if not self._main_period:            
            if 'orbit' in self.params:
                period = self.params['orbit']['period']
                t0 = self.params['orbit']['t0']
                shift = self.params['orbit']['phshift']
            elif len(self.bodies):
                period, t0, shift = self.bodies[0].get_period()
        else:
            period, t0, shift = super(BodyBag, self).get_period(self)
       
        return period, t0, shift
            
    
    def to_string(self,only_adjustable=False):
        """
        Human readable string representation of a BodyBag.
        
        @return: string representation
        @rtype: str
        """
        #"----------- BodyBag contains %d bodies----------- "%(len(self.bodies))
        txt = super(BodyBag,self).to_string(only_adjustable=only_adjustable)
        for i,body in enumerate(self.bodies):
            #"----------- Body %d ----------- "%(i)
            txt += body.to_string(only_adjustable=only_adjustable)
        return txt
    
    def reset(self):
        """
        We need to reimplement reset here.
        """
        for body in self.bodies:
            body.reset()
    
    def remove_ref(self, ref=None):
        """
        We need to reimplement remove_ref here.
        """
        # remove here
        super(BodyBag, self).remove_ref(ref=ref)
        # remove down
        for body in self.bodies:
            body.remove_ref(ref=ref)

    
    def change_ref(self, from_, to_):
        """
        We need to reimplement change_ref here.
        """
        # change here
        super(BodyBag, self).change_ref(from_, to_)
        for body in self.bodies:
            body.change_ref(from_, to_)

    
    def append(self,other):
        """
        Append a new Body to the BodyBag.
        """
        self.bodies.append(other)
    
    def get_mesh(self):
        """
        Return the mesh as a concatenated array of all mesh arrays.
        
        @return: combined mesh of all bodies
        @rtype: recarray
        """
        try:
            return np.hstack([b.mesh for b in self.bodies])
        except TypeError:
            #names = set(self.bodies[0].mesh.dtype.names)
            #for b in self.bodies[1:]:
                #names = names & set(b.mesh.dtype.names)
            #return np.hstack([b.mesh[list(names)] for b in self.bodies])
            msg = "Cannot merge meshes from different bodies. Make sure the dependable ParameterSet have the same labels in all bodies: i.e. if a light curve with label x is present in one Body, it needs to be there for all of them."
            myset = set(self.bodies[0].mesh.dtype.names)
            for b in self.bodies[1:]:
                msg += '\n'+ str(myset-set(b.mesh.dtype.names))
                msg += '\n <--> '+ str(set(b.mesh.dtype.names)-myset)
            raise TypeError(msg)
    
    def set_mesh(self,new_mesh):
        """
        Set the meshes of each body in the bag.
        """
        sizes = np.cumsum([0]+[len(b.mesh) for b in self.bodies])
        for i in range(len(self.bodies)):
            self.bodies[i].mesh = new_mesh[sizes[i]:sizes[i+1]]
    
    #def get_parset(self,*args,**kwargs):
        #"""
        #Returns the parameterSet and label of only the first Body.
        #"""
        #return self.bodies[0].get_parset(*args,**kwargs)
    
    def set_time(self, time, *args, **kwargs):
        """
        Set the time of all the Bodies in the BodyBag.
        
        We can dispatch corrected times to the separate bodies here if we want
        to correct for Roemer delays and stuff in hierarchical systems. Then
        we need to first come up with something that retrieves the orbit for
        each component. Then we can pass that on to keplerorbit.
        """
        for body in self.bodies:
            body.set_time(time, *args, **kwargs)
        
        if 'orbit' in self.params:
            # If we need to put this bodybag in an orbit and we need to correct
            # for ltt's, we can only ask for one of it's members to return their
            # proper time. But if you do this you assume the proper time of the
            # members of the bodybag is the same anyway. So this should be fine!
            #time = self.get_proper_time(time)[0]
            #-- once we have the mesh, we need to place it into orbit
            #keplerorbit.place_in_binary_orbit(self, time)
            n_comp = self.get_component()
            component = ('primary','secondary')[n_comp]
            orbit = self.params['orbit']
            loc, velo, euler = keplerorbit.get_binary_orbit(time,orbit, component)
            self.rotate_and_translate(loc=loc,los=(0,0,+1),incremental=True)
            mesh = self.mesh
            mesh['velo___bol_'] = mesh['velo___bol_'] + velo
            self.mesh = mesh
    

    
    def get_mass(self):
        """
        Return the total mass of the BodyBag.
        
        Any members that do not implement "get_mass" are assumed to have mass
        equal to zero.
        
        @param return: mass (Msol)
        @type return: float
        """
        total_mass = 0.
        for body in self.bodies:
            if hasattr(body, 'get_mass'):
                total_mass += body.get_mass()
        return total_mass
    
    #def get_distance(self):
        #globals_parset = self.get_globals()
        #if globals_parset is not None:
            #distance = globals_parset.request_value('distance', 'Rsol')
        #else:
            #distance = 10*constants.pc/constants.Rsol
        #return distance
        
    
    def set_label(self,label):
        """
        Set label of the BodyBag.
        """
        try:
            comp = self.get_component()
            if comp==0:
                self.params['orbit']['c1label'] = label
            elif comp==1:
                self.params['orbit']['c2label'] = label
        except Exception as msg:
            logger.debug(str(msg))
        
        self.label = label
            
    
    def get_label(self):
        if self.label is None and len(self.bodies)==1:
            return self.bodies[0].get_label()
        elif self.label is None:
            raise ValueError("BodyBag has no label and consists of more than one Bodies")
        else:
            return self.label
   
    
    def get_component(self):
        """
        Check which component this is.
        
        @return: 0 (primary) or 1 (secondary) or None (fail)
        @rtype: integer/None
        """
        try:
            if self.get_label()==self.params['orbit']['c1label']:
                return 0
            elif self.get_label()==self.params['orbit']['c2label']:
                return 1
            else:
                return None
        except TypeError:
            raise TypeError("No components (c1label,c2label) in the orbit parameter set")
        except KeyError:
            raise KeyError(("The BodyBag is not in a binary system, "
                            "perhaps some of the members are?"))
    
    def get_synthetic(self, *args, **kwargs):
        """
        Retrieve results from synthetic calculations.
        
        If C{cumulative=False}, results will be nested. Otherwise, they will be
        merged.
        """
        cumulative = kwargs.get('cumulative', True)
        
        # Sometimes synthetics can be added directly to the BodyBag, instead of
        # being built from the ones in the bodies list. 
        if kwargs.get('category', 'lc') in ['rv']:
            total_results = self.get_parset(ref=kwargs.get('ref',0), type='syn',
                                        category=kwargs.get('category', None))[0]
        
            if total_results is not None:
                return total_results
        
        # Prepare to return all the results
        total_results = []
        
        # Run over all bodies and get the synthetic stuff from there
        for i, body in enumerate(self.bodies):
            out = body.get_synthetic(*args, **kwargs)
            if out is not None:
                total_results.append(out)
        
        # Add the results together if cumulative results are required
        if cumulative is True and total_results:
            try:
                total_results = sum(total_results)
            except TypeError:
                total_results = None
        
        # If no results at all, replace with None for consistency with Body
        if total_results == []:
            total_results = None
        
        return total_results
    
    def get_obs(self,category='lc',ref=0):
        """
        Retrieve obs.
        """
        base,ref = self.get_parset(ref=ref,type='obs',category=category)
        return base
    
    def get_lc(self,ref=None):
        """
        Return the total light curve.
        
        All fluxes will be recursively summed up.
        """
        results = self.get_synthetic(type='lcsyn',ref=ref,cumulative=True)
        times = np.array(results['time'])
        signal = np.array(results['flux'])
        return times,signal

    def clear_synthetic(self,*args,**kwargs):
        super(BodyBag,self).clear_synthetic(*args,**kwargs)
        for body in self.bodies:
            body.clear_synthetic(*args,**kwargs)
    
    
    def get_model(self):
        mu,sigma,model = super(BodyBag,self).get_model()
        for body in self.bodies:
            mu_,sigma_,model_ = body.get_model()
            mu = np.hstack([mu,mu_])
            sigma = np.hstack([sigma,sigma_])
            model = np.hstack([model,model_])
        return mu,sigma,model
    
    
    def as_point_source(self,only_coords=False):
        coords = self.mesh['center'].mean(axis=0)
        #if 'orbit' in self.params:
            #distance = self.params['orbit'].request_value('distance','Rsol')
        #elif 'orbit' in self.bodies[0].params:
            #distance = self.bodies[0].params['orbit'].request_value('distance','Rsol')
        #else:
            #distance = 0
            #logger.warning("Don't know distance")
        globs = self.get_globals()
        if globs is not None:
            distance = globs.request_value('distance','Rsol')
        else:
            distance = 0.0
            
        coords[2] += distance
        if only_coords:
            return coords
        else:
            return dict(coordinates=coords)
            
    def get_barycenter(self):
        """
        Compute the barycenter of all children bodies (at the current time) 
        from the mesh and masses of the objects
        
        WARNING: still needs thorough testing   
        """
        
        params = [body.params['star'] if 'star' in body.params.keys() else body.params['component'] for body in self.bodies]
        distances = np.array([body.get_barycenter() for body in self.bodies])
        masses = np.array([body.get_value('mass') if 'mass' in ps.keys() else body.get_mass() for body,ps in zip(self.bodies,params)])

        return np.average(distances, weights=masses)
            
   
    #@decorators.parse_ref
    #def pl(self,wavelengths=None,ref='allpldep',sigma=5.,depth=0.4,time=None):
        #for lbl in ref:
            #if 'obs' in self.params and 'plobs' in self.params['obs'] and ref in self.params['obs']['plobs']:
                #wavelengths = 
            #else:
                #for body in self.bodies:
                    #body.pl(wavelengths=wavelengths,ref=lbl,sigma=sigma,depth=depth,time=time)
        
    #@decorators.parse_ref
    #def ifm(self,ref='allifdep',time=None):
        #"""
        #You can only do this if you have observations attached.
        #"""
        ##-- don't bother if we cannot do anything...
        #if hasattr(self,'params') and 'obs' in self.params:
            #if not ('ifobs' in self.params['obs']): return None
            #for lbl in ref:
                #ifobs,lbl = self.get_parset(type='obs',ref=lbl)
                #times = ifobs['time']
                #posangle = np.arctan2(ifobs['vcoord'],ifobs['ucoord'])/pi*180.
                #baseline = sqrt(ifobs['ucoord']**2 + ifobs['vcoord']**2) 
                #eff_wave = None if (not 'eff_wave' in ifobs or not len(ifobs['eff_wave'])) else ifobs['eff_wave']
                #keep = np.abs(times-time)<1e-8
                #output = observatory.ifm(self,posangle=posangle[keep],
                                     #baseline=baseline[keep],eff_wave=eff_wave,
                                     #ref=lbl,keepfig=False)
                                     ##ref=lbl,keepfig=('pionier_time_{:.8f}'.format(time)).replace('.','_'))
                #ifsyn,lbl = self.get_parset(type='syn',ref=lbl)
                #ifsyn['time'] += [time]*len(output[0])
                #ifsyn['ucoord'] += list(ifobs['ucoord'][keep])
                #ifsyn['vcoord'] += list(ifobs['vcoord'][keep])
                #ifsyn['vis'] += list(output[3])
                #ifsyn['phase'] += list(output[4])
        
    
    mesh = property(get_mesh,set_mesh)

    
class BinaryBag(BodyBag):
    """
    Convenience class for making a binary out of non-binary bodies.
        
    You can use it to make a binary out of one object, or two objects.
    
    Note: some stuff needs to be set automatically, like the mass ratio q.
    """
    def __new__(self, objs, orbit, solve_problems=True, **kwargs):
        """
        Parameter objs needs to a list, perhaps [None, object]  or [object, None]
        if you only want to create a BinaryBag of one object.
        
        To do: if one of the components is a star, optionally morph it to
        BinaryRocheStar
        """
        if len(objs)>2:
            raise ValueError("Binaries consist of a maximum of two objects ({} given)".format(len(objs)))
        
        system = []
        
        # Usually, we'll pick the label from the orbit to give to the BodyBag
        # The user can override this when 'label' is given in the kwargs
        kwargs.setdefault('label', orbit['label'])
        # But then we also need to update the orbital label
        orbit['label'] = kwargs.get('label')
        
        for i,iobject in enumerate(objs):
            if iobject is not None:
                # if the object has no "params" attribute, assume it is some
                # kind of list and then put all elements in a BodyBag
                if not isinstance(iobject,Body):
                    iobject = BodyBag(iobject,orbit=orbit,solve_problems=solve_problems)
                    ilabel = uuid.uuid4()
                    iobject.set_label(ilabel)
                    logger.info('BinaryBag: packed component {} in a BodyBag'.format(i))
                else:
                    try:
                        ilabel = iobject.get_label()
                    except ValueError:
                        ilabel = orbit['c{}label'.format(i+1)]
                        iobject.set_label(ilabel)
                        logger.info("Solved previous error: label set to {} (taken from orbit)".format(ilabel))
                        #ilabel = uuid.uuid4()
                #-- check if the Body is already in this orbit, or has an
                #   empty  orbit
                is_already_in_orbit = 'orbit' in iobject.params and iobject.params['orbit'] is orbit
                has_empty_orbit = 'orbit' in iobject.params and (iobject.params['orbit'] is None)
                # if it is already in this orbit, make sure it has the right
                # label
                if is_already_in_orbit:
                    logger.info("BinaryBag: Component {} (label={}) is already in this system, leaving as is".format(i,ilabel))
                    iobject.set_label(ilabel)    
                # if the object has an empty orbit, assume we have to fill it in now
                elif has_empty_orbit:
                    iobject.params['orbit'] = orbit
                    iobject.set_label(ilabel, component=i)
                    logger.info("BinaryBag: Component {} (label={}) had empty orbit".format(i,ilabel))
                # else, the object could have an orbit but it is not equal to
                # the one given. In that case, pack it into a BodyBag with the
                # given orbit.
                else:
                    logger.info("BinaryBag: Component {} (label={}) is normal Body".format(i,ilabel))
                    iobject = BodyBag([iobject],orbit=orbit)
                    iobject.set_label(ilabel)
                # Now set the label in the orbit parameterSet.
                if i==0:
                    orbit['c1label'] = ilabel
                else:
                    orbit['c2label'] = ilabel
                system.append(iobject)
        
        #-- pack in one system, but only if really necessary
        if len(system)>1:
            return BodyBag(system,solve_problems=solve_problems, **kwargs)
        else:
            return system[0]
        

class AccretionDisk(PhysicalBody):
    """
    Flaring Accretion Disk.
    
    The implementation of the L{AccretionDisk} follows closely the descriptions
    presented in the papers of [Copperwheat2010]_ and [Wood1992]_.
    
     <http://adsabs.harvard.edu/abs/2010MNRAS.402.1824C>
     <http://adsabs.harvard.edu/abs/1992ApJ...393..729W>
    
    There is no limb-darkening included yet.
    
    Flickering:
    
    http://iopscience.iop.org/0004-637X/486/1/388/fulltext/
    
    Brehmstrahlung:
    
    http://www.google.be/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&ved=0CDcQFjAB&url=http%3A%2F%2Fwww.springer.com%2Fcda%2Fcontent%2Fdocument%2Fcda_downloaddocument%2F9783319006116-c2.pdf%3FSGWID%3D0-0-45-1398306-p175157874&ei=30JtUsDYCqqX1AWEk4GQCQ&usg=AFQjCNFImeW-EOIauLtmoNqrBw7voYJNRg&sig2=RdoxaZJletpNNkEeyiIm2w&bvm=bv.55123115,d.d2k
    """
    def __init__(self,accretion_disk, mesh=None, pbdep=None,reddening=None,**kwargs):
        """
        Initialize a flaring accretion disk.
        
        The only parameters need for the moment are the disk parameters.
        In the future, also the details on the mesh creation should be given.
        """
        # Basic initialisation
        super(AccretionDisk,self).__init__(dim=3,**kwargs)
        
        # Prepare basic parameterSets and Ordered dictionaries
        check_input_ps(self, accretion_disk, ['accretion_disk'], 1)
        check_input_ps(self, mesh, ['mesh:disk'], 2)
        
        self.params['disk'] = accretion_disk
        if mesh is None:
            mesh = parameters.ParameterSet('mesh:disk')
        self.params['mesh'] = mesh
        self.params['pbdep'] = OrderedDict()
        #-- add interstellar reddening
        if reddening is not None:
            self.params['reddening'] = reddening
        #-- add the parameters to compute dependables
        if pbdep is not None:
            _parse_pbdeps(self,pbdep)
    
    def set_label(self,label):
        self.params['disk']['label'] = label
    
    def get_label(self):
        """
        Get the label of the Body.
        """
        return self.params['disk']['label']

    def compute_mesh(self):
        angular = self.params['mesh'].get_value('longit')
        radial = self.params['mesh'].get_value('radial')
        Rin = self.params['disk'].get_value('rin','Rsol')
        Rout = self.params['disk'].get_value('rout','Rsol')
        height = self.params['disk'].get_value('height','Rsol')
        
        if not 'logg' in self.mesh.dtype.names:
            lds = [('ld___bol','f8',(Nld_law,)),('proj___bol','f8')]
            for pbdeptype in self.params['pbdep']:
                for ipbdep in self.params['pbdep'][pbdeptype]:
                    ipbdep = self.params['pbdep'][pbdeptype][ipbdep]
                    lds.append(('ld_{0}'.format(ipbdep['ref']),'f8',(Nld_law,)))
                    lds.append(('proj_{0}'.format(ipbdep['ref']),'f8'))
                    #lds.append(('velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
                    #lds.append(('_o_velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
            dtypes = np.dtype(self.mesh.dtype.descr + \
                     lds + [('logg','f8'),('teff','f8'),('abun','f8')])
        else:
            dtypes = self.mesh.dtype
        N = (radial-1)*(angular-1)*2*2 + 4*angular-4
        self.mesh = np.zeros(N,dtype=dtypes)
        c = 0
        rs,thetas = np.linspace(Rin,Rout,radial),np.linspace(0,2*pi,angular)
        for i,r in enumerate(rs[:-1]):
            for j,th in enumerate(thetas[:-1]):
                self.mesh['_o_triangle'][c,0] = rs[i+1]*np.cos(thetas[j])
                self.mesh['_o_triangle'][c,1] = rs[i+1]*np.sin(thetas[j])
                self.mesh['_o_triangle'][c,2] = -height*rs[i+1]**1.5
                self.mesh['_o_triangle'][c,3] = rs[i]*np.cos(thetas[j])
                self.mesh['_o_triangle'][c,4] = rs[i]*np.sin(thetas[j])
                self.mesh['_o_triangle'][c,5] = -height*rs[i]**1.5
                self.mesh['_o_triangle'][c,6] = rs[i]*np.cos(thetas[j+1])
                self.mesh['_o_triangle'][c,7] = rs[i]*np.sin(thetas[j+1])
                self.mesh['_o_triangle'][c,8] = self.mesh['_o_triangle'][c,5]#-height*rs[i]**1.5
                
                self.mesh['_o_triangle'][c+1,0] = self.mesh['_o_triangle'][c,3]#rs[i]*np.cos(thetas[j])
                self.mesh['_o_triangle'][c+1,1] = self.mesh['_o_triangle'][c,4]#rs[i]*np.sin(thetas[j])
                self.mesh['_o_triangle'][c+1,2] = -self.mesh['_o_triangle'][c,5]#height*rs[i]**1.5
                self.mesh['_o_triangle'][c+1,3] = self.mesh['_o_triangle'][c,0]#rs[i+1]*np.cos(thetas[j])
                self.mesh['_o_triangle'][c+1,4] = self.mesh['_o_triangle'][c,1]#rs[i+1]*np.sin(thetas[j])
                self.mesh['_o_triangle'][c+1,5] = -self.mesh['_o_triangle'][c,2]#height*rs[i+1]**1.5
                self.mesh['_o_triangle'][c+1,6] = self.mesh['_o_triangle'][c,6]#rs[i]*np.cos(thetas[j+1])
                self.mesh['_o_triangle'][c+1,7] = self.mesh['_o_triangle'][c,7]#rs[i]*np.sin(thetas[j+1])
                self.mesh['_o_triangle'][c+1,8] = -self.mesh['_o_triangle'][c,8]#height*rs[i]**1.5
                
                self.mesh['_o_triangle'][c+2,0] = self.mesh['_o_triangle'][c,0]#rs[i+1]*np.cos(thetas[j])
                self.mesh['_o_triangle'][c+2,1] = self.mesh['_o_triangle'][c,1]#rs[i+1]*np.sin(thetas[j])
                self.mesh['_o_triangle'][c+2,2] = self.mesh['_o_triangle'][c+1,5]# height*rs[i+1]**1.5
                self.mesh['_o_triangle'][c+2,3] = rs[i+1]*np.cos(thetas[j+1])
                self.mesh['_o_triangle'][c+2,4] = rs[i+1]*np.sin(thetas[j+1])
                self.mesh['_o_triangle'][c+2,5] = self.mesh['_o_triangle'][c+1,5]#height*rs[i+1]**1.5
                self.mesh['_o_triangle'][c+2,6] = self.mesh['_o_triangle'][c,6]#rs[i]*np.cos(thetas[j+1])
                self.mesh['_o_triangle'][c+2,7] = self.mesh['_o_triangle'][c,7]#rs[i]*np.sin(thetas[j+1])
                self.mesh['_o_triangle'][c+2,8] = -self.mesh['_o_triangle'][c,8]#height*rs[i]**1.5
                
                self.mesh['_o_triangle'][c+3,0] = self.mesh['_o_triangle'][c+2,3]#rs[i+1]*np.cos(thetas[j+1])
                self.mesh['_o_triangle'][c+3,1] = self.mesh['_o_triangle'][c+2,4]#rs[i+1]*np.sin(thetas[j+1])
                self.mesh['_o_triangle'][c+3,2] = self.mesh['_o_triangle'][c,2]#-height*rs[i+1]**1.5
                self.mesh['_o_triangle'][c+3,3] = self.mesh['_o_triangle'][c,0]#rs[i+1]*np.cos(thetas[j])
                self.mesh['_o_triangle'][c+3,4] = self.mesh['_o_triangle'][c,1]#rs[i+1]*np.sin(thetas[j])
                self.mesh['_o_triangle'][c+3,5] = self.mesh['_o_triangle'][c,2]#-height*rs[i+1]**1.5
                self.mesh['_o_triangle'][c+3,6] = self.mesh['_o_triangle'][c,6]#rs[i]*np.cos(thetas[j+1])
                self.mesh['_o_triangle'][c+3,7] = self.mesh['_o_triangle'][c,7]#rs[i]*np.sin(thetas[j+1])
                self.mesh['_o_triangle'][c+3,8] = self.mesh['_o_triangle'][c,5]#-height*rs[i]**1.5
                c+=4
        for i,th in enumerate(thetas[:-1]):
            R = Rin
            self.mesh['_o_triangle'][c,0] = R*np.cos(thetas[i])
            self.mesh['_o_triangle'][c,1] = R*np.sin(thetas[i])
            self.mesh['_o_triangle'][c,2] = height*Rin**1.5
            self.mesh['_o_triangle'][c,3] = R*np.cos(thetas[i+1])
            self.mesh['_o_triangle'][c,4] = R*np.sin(thetas[i+1])
            self.mesh['_o_triangle'][c,5] = self.mesh['_o_triangle'][c,2]#height*Rin**1.5
            self.mesh['_o_triangle'][c,6] = self.mesh['_o_triangle'][c,0]#R*np.cos(thetas[i])
            self.mesh['_o_triangle'][c,7] = self.mesh['_o_triangle'][c,1]#R*np.sin(thetas[i])
            self.mesh['_o_triangle'][c,8] = -self.mesh['_o_triangle'][c,2]#-height*R**1.5
            
            self.mesh['_o_triangle'][c+1,0] = self.mesh['_o_triangle'][c,3]#R*np.cos(thetas[i+1])
            self.mesh['_o_triangle'][c+1,1] = self.mesh['_o_triangle'][c,4]#R*np.sin(thetas[i+1])
            self.mesh['_o_triangle'][c+1,2] = self.mesh['_o_triangle'][c,8]#-height*R**1.5
            self.mesh['_o_triangle'][c+1,3] = self.mesh['_o_triangle'][c,0]#R*np.cos(thetas[i])
            self.mesh['_o_triangle'][c+1,4] = self.mesh['_o_triangle'][c,1]#R*np.sin(thetas[i])
            self.mesh['_o_triangle'][c+1,5] = self.mesh['_o_triangle'][c,8]#-height*R**1.5
            self.mesh['_o_triangle'][c+1,6] = self.mesh['_o_triangle'][c,3]#R*np.cos(thetas[i+1])
            self.mesh['_o_triangle'][c+1,7] = self.mesh['_o_triangle'][c,4]#R*np.sin(thetas[i+1])
            self.mesh['_o_triangle'][c+1,8] = self.mesh['_o_triangle'][c,2]#+height*R**1.5
                
            c+=2
            R = Rout
            self.mesh['_o_triangle'][c,0] = R*np.cos(thetas[i])
            self.mesh['_o_triangle'][c,1] = R*np.sin(thetas[i])
            self.mesh['_o_triangle'][c,2] = -height*R**1.5
            self.mesh['_o_triangle'][c,3] = R*np.cos(thetas[i+1])
            self.mesh['_o_triangle'][c,4] = R*np.sin(thetas[i+1])
            self.mesh['_o_triangle'][c,5] = -height*R**1.5
            self.mesh['_o_triangle'][c,6] = R*np.cos(thetas[i])
            self.mesh['_o_triangle'][c,7] = R*np.sin(thetas[i])
            self.mesh['_o_triangle'][c,8] = +height*R**1.5
            
            self.mesh['_o_triangle'][c+1,3] = R*np.cos(thetas[i])
            self.mesh['_o_triangle'][c+1,4] = R*np.sin(thetas[i])
            self.mesh['_o_triangle'][c+1,5] = height*R**1.5
            self.mesh['_o_triangle'][c+1,0] = R*np.cos(thetas[i+1])
            self.mesh['_o_triangle'][c+1,1] = R*np.sin(thetas[i+1])
            self.mesh['_o_triangle'][c+1,2] = height*R**1.5
            self.mesh['_o_triangle'][c+1,6] = R*np.cos(thetas[i+1])
            self.mesh['_o_triangle'][c+1,7] = R*np.sin(thetas[i+1])
            self.mesh['_o_triangle'][c+1,8] = -height*R**1.5
                
            c+=2
 
        self.mesh['triangle'] = self.mesh['_o_triangle']
        self.compute_centers()
        self.compute_sizes()
        self.compute_normals()
        #self.rotate(incl=45.,Omega=1.)
        #self.rotate()
        self.rotate_and_translate()
        self.detect_eclipse_horizon(eclipse_detection='hierarchical')        
    
    def surface_gravity(self):
        """
        Calculate the local surface gravity:
        
        g = GMhost * z / R**3 (see e.g. Diaz et al. 1996)
        """
        r = coordinates.norm(self.mesh['_o_center'],axis=1)*constants.Rsol
        z = self.mesh['_o_center'][:,2]*constants.Rsol
        M_wd = self.params['disk'].get_value('mass','kg')
        g = constants.GG*M_wd*np.abs(z)/r**3
        self.mesh['logg'] = np.log10(g*100) # logg in cgs
        
    
    def temperature(self,time=None):
        r"""
        Temperature is calculated according to a scaling law.
        
        One model is inpsired upon [Wood1992]_ and [Copperwheat2010]_ 
        
        .. math::
        
            T_\mathrm{eff}(r)^4 = (G M_\mathrm{wd} \dot{M}) / (8\pi r^3) (1-b\sqrt{R_\mathrm{in}/r}) 
        
        An alternative formula might be from [Frank1992]_ (p. 78):
        
        .. math::
            T_\mathrm{eff}(r)^4 = \frac{W^{0.25} 3G M_\mathrm{wd} \dot{M}}{8 \pi r^3 \sigma}  (1-\sqrt{\frac{R_*}{r}}) \left(\frac{r}{R_*}\right)^{0.25\beta}
        
        with :math:`\beta=-0.75` and :math:`W=1.0` for the standard model.
        """
        r = coordinates.norm(self.mesh['_o_center'],axis=1)*constants.Rsol
        context = self.params['disk'].get_context().split(':')
        
        # copperwheat
        if len(context)==1:
            disk_type = 'copperwheat'
        else:
            disk_type = context[1]
            
        if disk_type == 'copperwheat':    
            Mdot = self.params['disk'].get_value('dmdt','kg/s')
            M_wd = self.params['disk'].get_value('mass','kg')
            Rin = self.params['disk'].get_value('rin','m')
            b = self.params['disk']['b']
            sigTeff4 = constants.GG*M_wd*Mdot/(8*pi*r**3)*(1-b*sqrt(Rin/r))
            sigTeff4[sigTeff4<0] = 1e-1 #-- numerical rounding (?) can do weird things
            self.mesh['teff'] = (sigTeff4/constants.sigma)**0.25
        
        elif disk_type == 'frank':
            Mdot = self.params['disk'].get_value('dmdt','kg/s')
            M_wd = self.params['disk'].get_value('mass','kg')
            Rin = self.params['disk'].get_value('rin','m')
            W = self.params['disk'].get_value('W')
            raise NotImplementedError
        
    @decorators.parse_ref
    def intensity(self, *args, **kwargs):
        """
        Calculate local intensity and limb darkening coefficients.
        """
        ref = kwargs.pop('ref',['all'])
        beaming_alg = kwargs.get('beaming_alg', 'full')
        parset_isr = dict()#self.params['reddening']
        #-- now run over all labels and compute the intensities
        for iref in ref:
            parset_pbdep,ref = self.get_parset(ref=iref,type='pbdep')
            limbdark.local_intensity(self,parset_pbdep,parset_isr, beaming_alg=beaming_alg)
            
    def projected_intensity(self,los=[0.,0.,+1],ref=0,method='numerical',with_partial_as_half=True,
                            beaming_alg='none'):
        """
        Calculate local intensity of an AccretionDisk.
        """
        if method!='numerical':
            raise ValueError("Only numerical computation of projected intensity of AccretionDisk available")
        idep,ref = self.get_parset(ref=ref, type='pbdep')
        ld_func = idep['ld_func']
        proj_int = generic_projected_intensity(self, method=method,
                             ld_func=ld_func, ref=ref, los=los, 
                             with_partial_as_half=with_partial_as_half)
        
        return proj_int
    
    def set_time(self,time, ref='all', beaming_alg='none'):
        """
        Set the time of the AccretionDisk object
        """
        logger.info('===== SET TIME TO %.3f ====='%(time))
        if self.time is None:
            self.reset_mesh()
            
            self.compute_mesh()
            self.surface_gravity()
            self.temperature()
            self.intensity(beaming_alg=beaming_alg)
            
            incl = self.params['disk']['incl']/180.*np.pi
            Omega = self.params['disk']['long']/180.*np.pi
            self.rotate_and_translate(incl=incl, Omega=Omega, loc=(0,0,0),
                                      los=(0,0,+1),
                                      incremental=True)
        else:
            self.reset_mesh()
            self.temperature()
            Omega = self.params['disk']['long']/180.*np.pi
            incl = self.params['disk']['incl']/180.*np.pi
            self.rotate_and_translate(incl=incl, Omega=Omega, loc=(0,0,0),
                                      los=(0,0,+1),
                                      incremental=True)
            
        self.time = time
        
        
        
class Star(PhysicalBody):
    """
    Body representing a Star.
    
    A Star Body only has one obligatory ParameterSet: context ``star``. It sets
    the basic properties such as mass, radius and temperature. The ParameterSet
    also contains some atmosphere and limbdarkening parameters, which are all
    bolometric quantities.
    
    Optional parameterSets:
    
        - ``mesh``: determines mesh properties, such as mesh density and algorithm
        - ``reddening``: set interstellar reddening properties (law, extinction)
        - ``circ_spot``: spot properties (can be a list)
        - ``puls``: pulsation mode properties (can be a list)
        - ``magnetic_field``: global magnetic field properties (dipole...)
        - ``velocity_field``: surface velocity field properties (macroturbulence)
        - ``granulation``: surface granulation
        - ``globals``: systemic velocity, position, distance...
        
    As for any other Body, you can give also a list of
    
        - ``pbdep``: passband dependables
        - ``obs``: observations
    
    Useful functions:
    
    .. autosummary::
    
        get_mass
        volume
        vsini
        critical_rotation
    
    **Example usage**:
    
    Construct a simple star with the default parameters representing the Sun
    (sort of, don't start nitpicking):
    
    >>> star_pars = parameters.ParameterSet('star', label='mystar')
    >>> lcdep1 = parameters.ParameterSet('lcdep', ref='mylc')
    >>> mesh = parameters.ParameterSet('mesh:marching')
    >>> star = Star(star_pars, mesh, pbdep=[lcdep1])
    
    We initialized the star with these parameters::
    
        >>> print(star_pars)
              teff 5777.0           K phoebe Effective temperature
            radius 1.0           Rsol phoebe Radius
              mass 1.0           Msol phoebe Stellar mass
               atm blackbody       -- phoebe Bolometric Atmosphere model
         rotperiod 22.0             d phoebe Equatorial rotation period
             gravb 1.0             -- phoebe Bolometric gravity brightening
              incl 90.0           deg phoebe Inclination angle
           surface roche           -- phoebe Type of surface
        irradiator False           -- phoebe Treat body as irradiator of other objects
             label mystar          -- phoebe Name of the body
           ld_func uniform         -- phoebe Bolometric limb darkening model
         ld_coeffs [1.0]           -- phoebe Bolometric limb darkening coefficients
          surfgrav 274.351532944  n/a constr constants.GG*{mass}/{radius}**2    
    
    Upon initialisation, only the basic parameters are set, nothing is computed.
    This is fine for passing Stars on to automatic computing functions like
    :py:func:`observatory.compute <phoebe.backend.observatory.compute>`. If you
    want to do some manual work with the Star class, you probably first need
    to compute the mesh. This can be done at the most basic level with the
    function :py:func:`Star.compute_mesh`, but it is easiest just to set the
    time via :py:func:`Star.set_time`, and let the Body take care of itself:
    
    >>> star.set_time(0)
    
    Now we can make plots:
    
    >>> p = mlab.figure(bgcolor=(0.5,0.5,0.5))
    >>> p = star.plot3D(select='mu',colormap='spectral')
    >>> p = mlab.colorbar()
    
    ]include figure]]images/universe_star_0001.png]
    
    >>> out = observatory.image(star,savefig=False)
    
    ]include figure]]images/universe_star_0002.png]
    
    We can do the same for a Sun-like star which is rotation close to its
    critical rotation velocity:
    
    >>> star_pars['rotperiod'] = 0.25#0.213,'d'#0.21285,'d'
    >>> star_pars['gravb'] = 0.25
    >>> star = Star(star_pars,mesh,pbdep=[lcdep1])
    >>> star.set_time(0)
    
    Again with the plots of any quantity that has been set:
    
    >>> p = mlab.figure(bgcolor=(0.5,0.5,0.5))
    >>> p = star.plot3D(select='teff',colormap='spectral')
    >>> p = mlab.colorbar()
    
    ]include figure]]images/universe_star_0003.png]
    
    >>> out = observatory.image(star,savefig=False)
    
    ]include figure]]images/universe_star_0004.png]
    
    """
    _params_tree = dict(star=None, mesh=None, reddening=None,
                        puls='list', circ_spot='list', magnetic_field=None,
                        velocity_field='list', granulation='list',
                        position=None)
    
    def __init__(self, star=None, mesh=None, reddening=None, puls=None,
                 circ_spot=None, magnetic_field=None, velocity_field=None,
                 granulation=None, position=None, pbdep=None, obs=None,
                 label=None, **kwargs):
        """
        Initialize a Star.
        
        What needs to be done? Well, sit down and let me explain. Are you
        sitting down? Yes, then let's begin our journey through the birth
        stages of a Star. In many ways, the birth of a Star can be regarded as
        the growth of a human child. First of all, the basic properties
        need to be set. Properties of a star include physical properties,
        but also computational properties such as mesh density and algorithm.
        This is also the place where the details on dependables, data and
        results will be initialised.
        
        If we have information on spots and pulsations, we attach them to the
        root here too.
        """
        # Basic initialisation
        super(Star, self).__init__(dim=3)
        
        # If there is no star, create default one
        if star is None:
            star = parameters.ParameterSet('star')
        
        # If there is no mesh, create default one
        if mesh is None:
            mesh = parameters.ParameterSet('mesh:marching')
        
        # Prepare basic parameterSets and Ordered dictionaries
        check_input_ps(self, star, ['star'], 1)
        check_input_ps(self, mesh, ['mesh:marching', 'mesh:wd'], 2)
        
        self.params['star'] = star
        self.params['mesh'] = mesh
        self.params['pbdep'] = OrderedDict()
        self.params['obs'] = OrderedDict()
        self.params['syn'] = OrderedDict()
        
        # Shortcut to make a binaryStar
        #if 'orbit' in kwargs:
        #   myorbit = kwargs.pop('orbit')
        #    check_input_ps(self, myorbit, ['orbit'], 'orbit')
        #    self.params['orbit'] = myorbit
        
        # Add globals parameters, but only if given. DO NOT add default ones,
        # that can be confusing
        if position is not None:
            check_input_ps(self, position, ['position'], 'position')
            self.params['position'] = position
        
        # Add interstellar reddening (if none is given, set to the default, this
        # means no reddening
        if reddening is not None:
            check_input_ps(self, reddening, ['reddening:interstellar'], 'reddening')
            self.params['reddening'] = reddening
        
        # Add spot parameters when applicable
        if circ_spot is not None:
            if not isinstance(circ_spot, list):
                to_add = [circ_spot]
            else:
                to_add = circ_spot
            
            # Perhaps the user gave an empty list, then that's a bit silly
            if len(to_add) > 0:
                for ito_add in to_add:
                    check_input_ps(self, ito_add, ['circ_spot'], 'circ_spot', is_list=True)
                self.params['circ_spot'] = to_add
            
        # Add pulsation parameters when applicable
        if puls is not None:
            if not isinstance(puls, list):
                to_add = [puls]
            else:
                to_add = puls
            
            # Perhaps the user gave an empty list, then that's a bit silly
            if len(puls) > 0:
                for ito_add in to_add:
                    check_input_ps(self, ito_add, ['puls'], 'puls', is_list=True)
                self.params['puls'] = to_add
            
        # Add magnetic field parameters when applicable
        if magnetic_field is not None:
            check_input_ps(self, magnetic_field,
                          ['magnetic_field:dipole','magnetic_field:quadrupole'],
                           'magnetic_field')
            self.params['magnetic_field'] = magnetic_field
        
        # Add velocity field parameters when applicable
        if velocity_field is not None:
            if not isinstance(velocity_field, list):
                to_add = [velocity_field]
            else:
                to_add = velocity_field
            
            # Perhaps the user gave an empty list, then that's a bit silly
            if len(to_add) > 0:
                for ito_add in to_add:
                    check_input_ps(self, ito_add, ['velocity_field:turb',
                                                   'velocity_field:meri'],
                                   'velocity_field', is_list=True)
                self.params['velocity_field'] = to_add
        
        # Add granulation field parameters when applicable
        if granulation is not None:
            if not isinstance(granulation, list):
                to_add = [granulation]
            else:
                to_add = granulation
            
            # Perhaps the user gave an empty list, then that's a bit silly
            if len(to_add) > 0:
                for ito_add in to_add:
                    check_input_ps(self, ito_add, ['granulation'],
                                   'granulation', is_list=True)
                self.params['granulation'] = to_add
        
        # Add the parameters to compute dependables
        if pbdep is not None:
            _parse_pbdeps(self, pbdep)
        
        # Add the parameters from the observations
        if obs is not None:
            _parse_obs(self, obs)
        
        # The label should be given in the Star parameterSet, but this isn't
        # always very intuitive. We allow setting the label via an extra keyword
        # argument
        if label is not None:
            self.set_label(label)
        
        # Check for leftover kwargs and report to the user
        if kwargs:
            raise ValueError("Unused keyword arguments {} upon initialization of Star".format(kwargs.keys()))
        
        # Initialise the mesh
        init_mesh(self)
        
        # Generate a comprehensive log message, that explains what has been
        # added:
        msg = "Created Star {}".format(self.get_label())
        msg_ = []
        if circ_spot is not None:
            msg_.append('{} circular spots'.format(len(self.params['circ_spot'])))
        if puls is not None:
            msg_.append('{} pulsations'.format(len(self.params['puls'])))
        if magnetic_field is not None:
            msg_.append('a magnetic field')
        if 'pbdep' in self.params:
            for type in self.params['pbdep']:
                msg_.append('{} {}'.format(len(self.params['pbdep'][type]),type))
        if 'obs' in self.params:
            for type in self.params['obs']:
                msg_.append('{} {}'.format(len(self.params['obs'][type]),type))
        if len(msg_):
            msg = msg + ': ' + ', '.join(msg_)
        logger.info(msg)
        
                
    def set_label(self, label):
        """
        Set the label of a Star
        
        @param label: label of the Star
        @type label: str
        """
        self.params['star']['label'] = label

    
    def get_label(self):
        """
        Get the label of a Star.
        
        @return: label of the Star
        @rtype: str
        """
        return self.params['star']['label']
    
    
    def get_component(self):
        """
        Check which component this Star is.
        
        @return: 0 (primary) or 1 (secondary) or None (fail)
        @rtype: integer/None
        """
        #print "Calling get component from BinaryStar",self.label
        if 'orbit' in self.params:
            if self.get_label()==self.params['orbit']['c1label']:
                return 0
            elif self.get_label()==self.params['orbit']['c2label']:
                return 1
        return None
        
    def get_mass(self):
        """
        Return the mass of a star.
        
        @param return: mass (Msol)
        @type return: float
        """
        return self.params['star']['mass']
    
    def get_period(self):
        """
        Extract the main period from the star.
        
        Overrides default from Body.
        """
        if not self._main_period:
            if 'puls' in self.params:
                period = 1.0/self.params['puls'][0]['freq']
                t0 = self.params['puls'][0]['t0']
                shift = 0.0
            else: 
                period = self.params['star']['rotperiod']
                t0 = 0.0
                shift = 0.0
        else:
            period, t0, shift = super(Star, self).get_period()
            
        return period, t0, shift
    
    
    def volume(self):
        """
        Compute volume of the convex mesh of the Star.
        """
        norm = coordinates.norm(self.mesh['_o_center'],axis=1)
        return np.sum(self.mesh['_o_size']*norm/3.)
    
    def surface_gravity(self):
        """
        Calculate local surface gravity of a Star.
        """
        # Retrieve basic information on coordinates, mass etc..
        x, y, z = self.mesh['_o_center'].T
        r = sqrt(x**2 + y**2 + z**2)
        M = self.params['star'].request_value('mass', 'kg')
        rp = self.params['star'].request_value('r_pole', 'Rsol')
        
        # Some coordinate transformations
        sin_theta = sqrt(x**2 + y**2) / r
        cos_theta = z / r
        X = r / rp
        rp = conversions.convert('Rsol', 'm', rp)
        
        # Information on rotation
        Omega = self.params['star'].request_value('Omega_rot')
        omega = 2*pi/self.params['star'].request_value('rotperiod', 's')
        if self.params['star']['shape'] == 'sphere':
            omega = 0.
            Omega = 0.
        
        # Compute local surface gravity
        r_ = r * constants.Rsol
        grav_r = -constants.GG*M / r_**2 + r_ * (omega*sin_theta)**2
        grav_th = r_*omega**2 * sin_theta * cos_theta
        local_grav = sqrt(grav_r**2 + grav_th**2)
        
        # Convert suface gravity from m/s2 to [cm/s2]
        self.mesh['logg'] = np.log10(local_grav) + 2.0
        logger.info("derived surface gravity: %.3f <= log g<= %.3f (Rp=%s)"%(self.mesh['logg'].min(),self.mesh['logg'].max(),rp/constants.Rsol))
    
    
    def temperature(self, time):
        """
        Calculate local temperature of a Star.
        """
        # If the gravity brightening law is not specified, use 'Zeipel's
        gravblaw = self.params['star'].get('gravblaw', 'zeipel')
        
        # If we're using Claret's gravity darkening coefficients, we're actually
        # using von Zeipel's law but with interpolated coefficients
        if gravblaw == 'claret':
            teff = np.log10(self.params['star']['teff'])
            logg = np.log10(self.params['star'].request_value('g_pole')*100)
            abun = self.params['star']['abun']
            axv, pix = roche.claret_gravb()
            gravb = interp_nDgrid.interpolate([[teff], [logg], [abun]], axv, pix)[0][0]
            logger.info('gravb(Claret): teff = {:.3f}, logg = {:.6f}, abun = {:.3f} ---> gravb = {:.3f}'.format(10**teff, logg, abun, gravb))            
            self.params['star']['gravb'] = gravb
            gravblaw = 'zeipel'
            
        # Compute the temperature
        getattr(roche,'temperature_{}'.format(gravblaw))(self)
        
        # Perhaps we want to add spots.
        self.add_spots(time)
    
    
    def abundance(self, time=None):
        """
        Set the abundance of a Star.
        """
        self.mesh['abun'] = list(self.params.values())[0]['abun']
    
    
    def magnetic_field(self, time=None):
        """
        Calculate the magnetic field in a Star.
        
        The magnetic field can be a :py:func:`dipole <phoebe.atmospheres.magfield.get_dipole>`
        or a :py:func:`(non-linear) quadrupole <phoebe.atmospheres.magfield.get_quadrupole>`
        """
        # Figure out if we have a dipole or quadrupole
        parset = self.params['magnetic_field']
        context = parset.get_context()
        topology = context.split(':')[-1]
        
        # Some basic quantities we need regardless of the topology
        Bpolar = parset.get_value('Bpolar')
        R = self.params.values()[0].get_value('radius')
        r_ = self.mesh['_o_center']
        
        # Then finally get the field according to its topology
        if topology == 'dipole':
            beta = parset.get_value('beta', 'rad')
            phi0 = parset.get_value('phi0', 'rad')
            B = magfield.get_dipole(time, r_, R, beta, phi0, Bpolar)
        
        elif topology == 'quadrupole':
            beta1 = parset.get_value('beta1', 'rad')
            phi01 = parset.get_value('phi01', 'rad')
            beta2 = parset.get_value('beta2', 'rad')
            phi02 = parset.get_value('phi02', 'rad')
            B = magfield.get_quadrupole(time, r_, R, beta1, phi01, beta2, phi02, Bpolar)
        
        # And add it to the mesh!
        self.mesh['_o_B_'] = B
        self.mesh['B_'] = self.mesh['_o_B_']
        
        logger.info("Added {} magnetic field with Bpolar={}G".format(topology, Bpolar))
        #logger.info("Maximum B-field on surface = {}G".format(coordinates.norm(B, axis=1).max()))
    
    
    @decorators.parse_ref
    def intensity(self, ref='all', beaming_alg='none'):
        """
        Calculate local intensity and limb darkening coefficients of a Star.
        """
        #-- now run over all labels and compute the intensities
        parset_isr = dict() #self.params['reddening']
        for iref in ref:
            parset_pbdep, ref = self.get_parset(ref=iref, type='pbdep')
            limbdark.local_intensity(self, parset_pbdep, parset_isr, beaming_alg=beaming_alg)
        
    
    @decorators.parse_ref
    def velocity(self, time=None, ref=None):
        """
        Calculate the velocity of each surface via the rotational velocity of a Star.
        """
        if time is None:
            time = self.time
        
        # Rotational velocity: first collect some information
        omega_rot = 1./self.params['star'].request_value('rotperiod','d')
        omega_rot = np.array([0.,0.,-omega_rot])
        logger.info('Calculating rotation velocity (Omega={:.3f} rad/d)'.format(omega_rot[-1]*2*pi))
        
        # Consistent shape and differential rotation
        if self.params['star']['shape']=='equipot' and self.params['star']['diffrot']!=0:
            #-- compute the local rotation velocities in cy/d
            b1,b2 = self.subdivision['mesh_args'][1:3]
            rpole_sol = self.subdivision['mesh_args'][-1]
            s = sqrt(self.mesh['_o_center'][:,0]**2+self.mesh['_o_center'][:,1]**2)/rpole_sol
            M = self.params['star'].request_value('mass','kg')
            r_pole = self.params['star'].request_value('radius','m')
            Omega_crit = sqrt( 8*constants.GG*M / (27.*r_pole**3))
            omega_rot = (b1+b2*s**2)/0.54433105395181736*Omega_crit/(2*pi)*3600*24.
            omega_rot = np.column_stack([np.zeros_like(omega_rot),\
                                   np.zeros_like(omega_rot),\
                                   -omega_rot])
            
        # Spherical shape but still differential rotation
        # OK, I realize this is extremely messy and slow code, but who's going
        # to use this seriously?
        elif self.params['star']['diffrot']!=0:
            M = self.params['star'].request_value('mass','kg')
            r_pole = self.params['star'].request_value('radius','m')
            Omega_crit = sqrt( 8*constants.GG*M / (27.*r_pole**3))
            diffrot = self.params['star'].get_value('diffrot','s')
            Period_eq = diffrot + self.params['star'].request_value('rotperiod','s')
            Omega_eq = 2*pi/Period_eq/Omega_crit
            Omega_param= 2*pi/self.params['star'].request_value('rotperiod','s')
            Omega = Omega_param/Omega_crit
            b1 = Omega*0.54433105395181736
            b2 = roche.diffrotlaw_to_internal(Omega,Omega_eq)
            rpole_sol = self.subdivision['mesh_args'][-1]
            s = sqrt(self.mesh['_o_center'][:,0]**2+self.mesh['_o_center'][:,1]**2)/rpole_sol
            omega_rot = (b1+b2*s**2)/0.54433105395181736*Omega_crit/(2*pi)*3600*24.
            omega_rot = np.column_stack([np.zeros_like(omega_rot),\
                                   np.zeros_like(omega_rot),\
                                   -omega_rot])
            
        # The velocity is the cross product of the centers with the rotation
        # vector pointed in the Z direction.
        velo_rot = 2*pi*np.cross(self.mesh['_o_center'],omega_rot) #NX3 array
        
        # We can add velocity fields here if we wish:
        if 'velocity_field' in self.params:
            for ps in self.params['velocity_field']:
                context = ps.get_context()
                subcontext = context.split(':')[1]
                
                # Macroturbulence
                if subcontext == 'turb':
                    vmacro_rad = ps.request_value('vmacro_rad', 'Rsol/d')
                    vmacro_tan = ps.request_value('vmacro_tan', 'Rsol/d')
                
                    if vmacro_rad > 0 or vmacro_tan > 0:
                        normal = self.mesh['_o_normal_']
                        vmacro = velofield.get_macroturbulence(normal,
                                                      vmacro_rad=vmacro_rad,
                                                      vmacro_tan=vmacro_tan)
                        velo_rot += vmacro
                        
                # Meridional circulation
                elif subcontext == 'meri':
                    radius = ps.request_value('location')
                    inner_radius = ps.request_value('bottom')
                    vmeri_ampl = ps.request_value('vmeri_ampl', 'Rsol/d')
                    wc = ps.request_value('penetration_depth')
                    angle = ps.request_value('latitude', 'rad')
                    if vmeri_ampl>0 and radius>inner_radius:
                        center = self.mesh['_o_center']
                        vmeri = velofield.get_meridional(center, radius,
                                                         inner_radius,
                                                         vmeri_ampl, wc,
                                                         angle)
                        velo_rot += vmeri
                    
            
        self.mesh['_o_velo___bol_'] = velo_rot
        self.mesh['velo___bol_'] = velo_rot
        
        
        
    
    def projected_intensity(self, los=[0.,0.,+1], ref=0, method=None,
                            with_partial_as_half=True, beaming_alg='none'):
        """
        Calculate projected intensity of a Star.
        
        We can speed this up if we compute the local intensity first, keep track of the limb darkening
        coefficients and evaluate for different angles. Then we only have to do a table lookup once.
        """
        idep, ref = self.get_parset(ref=ref, type='pbdep')
        
        if idep is None:
            raise ValueError("Unknown reference, use any of {} or __bol".format(", ".join(self.get_refs())))

        if method is None:
            method = 'method' in idep and idep['method'] or 'numerical'
        ld_func = idep['ld_func']
        l3 = idep.get('l3', 0.)
        pblum = idep.get('pblum', -1.0)
        
        # Compute projected intensity, and correct for the distance and
        # reddening
        proj_int = generic_projected_intensity(self, method=method,
                ld_func=ld_func, ref=ref, beaming_alg=beaming_alg,
                with_partial_as_half=with_partial_as_half)
            
        # Take passband luminosity into account
        if pblum >= 0:
            return proj_int*pblum + l3
        else:
            return proj_int + l3
        
    
    def projected_velocity(self,los=[0,0,+1],ref=0,method=None):
        rvdep,ref = self.get_parset(ref=ref,type='pbdep')
        ld_func = rvdep.request_value('ld_func')
        method = 'numerical'
        return limbdark.projected_velocity(self,method=method,ld_func=ld_func,ref=ref)
    
    
    def vsini(self,unit='km/s'):
        """
        Compute the vsini.
        """
        radius = self.params['star'].request_value('radius','km')
        period_pl = self.params['star'].request_value('rotperiod','s')
        diffrot = self.params['star'].request_value('diffrot','s')
        incl = self.params['star'].request_value('incl','rad')
        period_eq = period_pl + diffrot
        if self.params['star']['shape']=='equipot':
            mass = self.params['star'].request_value('mass','Msol')
            rpole = self.params['star'].request_value('radius','Rsol')
            omega_crit = roche.critical_angular_frequency(mass,rpole)
            omega_eq = 2*pi/period_eq/omega_crit
            omega_pl = 2*pi/period_pl/omega_crit
            b1 = omega_pl*0.54433105395181736
            b2 = roche.diffrotlaw_to_internal(omega_pl,omega_eq)
            r0 = -marching.projectOntoPotential(np.array((-0.02, 0.0, 0.0)), 'DiffRotateRoche', b1,b2,0,1.0).r[0]
            radius = r0*radius
        vsini = 2*pi*radius/period_eq*np.sin(incl)
        logger.info('Computed vsini = {} km/s'.format(vsini))
        return vsini
    
    
    def critical_rotation(self,frequency='Hz',period=None):
        """
        Return critical rotation frequency/period.
        """
        M = self.params['star'].request_value('mass','kg')
        R = self.params['star'].request_value('radius','m')
        Omega_crit = sqrt( 8*constants.GG*M / (27.*R**3))
        if period is not None:
            if not isinstance(period,str):
                period = 's'
                return conversions.convert('s',period,2*pi/Omega_crit)
        #-- else, frequency in Hz or custom
        if not isinstance(frequency,str):
            frequency = 'Hz'
        return conversions.convert('rad/s',frequency,Omega_crit)
        
    
    
    def add_spots(self,time):
        """
        Adjust the local properties for the presence of spots.
        
        The number of subdivisions is the maximum number of subdivisions for
        all spots. That is, we do not allow to ubdivide one spot two times
        and another one three times: reason: that would need some more
        implementation time.
        """
        if 'circ_spot' in self.params:
            max_subdiv = max([spot_pars['subdiv_num'] for spot_pars in self.params['circ_spot']])
            for j in range(max_subdiv+1):
                last_iter = (j==max_subdiv)
                for i,spot_pars in enumerate(self.params['circ_spot']):
                    logger.info('Spot {}'.format(i))
                    spots.add_circular_spot(self,time,spot_pars,update_temperature=last_iter)
                if not last_iter:
                    self.subdivide(subtype=2)
                
    
    def add_pulsations(self,time=None):
        """
        Add pulsations to a Star.
        """
        pulsations.add_pulsations(self, time=time)
    
    def add_granulation(self, time=None):
        """
        Add granulation to a Star.
        """
        if time is None:
            time = 0
            
        # Don't add granulation if there is none
        if not 'granulation' in self.params:
            return None
        
        for gran in self.params['granulation']:
            
            # Don't add granulation if there aren't any cells
            if gran['cells'] == 0:
                continue
            
            # Generate the Worley noise pattern
            values, velocity = spots.worley_noise(self, seed=int(time*1000),
                                                  metric=gran['pattern'],
                                                  feature_points=gran['cells'],
                                                  max_angle=gran['vgran_angle'])
            
            # Adapt local effective temperatures
            deltateff = values - np.median(values)
            deltateff = deltateff / np.std(deltateff) * gran['teff_ampl']
            self.mesh['teff'] = self.mesh['teff'] * (1 + deltateff/self.mesh['teff'])
            
            # Adapt local velocity fields
            velo = velocity * gran['vgran_ampl'] * kms_2_rsold
            self.mesh['_o_velo___bol_'] = self.mesh['_o_velo___bol_'] + velo
            self.mesh['velo___bol_'] = self.mesh['_o_velo___bol_']
        
        logger.info("Add granulation: {} < teff < {}".format(self.mesh['teff'].min(),
                                                           self.mesh['teff'].max()))
        
        
    
    def compute_mesh(self,time=None):
        """
        Compute the mesh of a Star.
        """
        M = self.params['star'].request_value('mass','kg')
        r_pole = self.params['star'].request_value('radius','m')
        r_pole_sol = self.params['star'].request_value('radius','Rsol')
        g_pole = constants.GG*M/r_pole**2
        Omega_crit = sqrt( 8*constants.GG*M / (27.*r_pole**3))
        Omega_param= 2*pi/self.params['star'].request_value('rotperiod','s')
        Omega = Omega_param/Omega_crit
        logger.info('rotation frequency (polar) = %.6f Omega_crit'%(Omega))
        
        self.params['star'].add_constraint('{{r_pole}} = {0:.16g}'.format(r_pole))
        self.params['star'].add_constraint('{{g_pole}} = {0:.16g}'.format(g_pole))
        self.params['star'].add_constraint('{{Omega_rot}} = {0:.16g}'.format(Omega))
        
        #-- check for sphere-approximation
        diffrot = 0.
        surface = 'RotateRoche'
        self.subdivision['mesh_args'] = surface,Omega,1.0,r_pole_sol
        if self.params['star']['shape']=='sphere':
            Omega = 0.
            self.subdivision['mesh_args'] = surface,Omega,1.0,r_pole_sol
            logger.info("using non-rotating surface approximation")
        #-- check for the presence of differential rotation
        elif 'diffrot' in self.params['star'] and self.params['star']['diffrot']!=0:
            #-- retrieve equatorial rotation period and convert to angular
            #   frequency
            diffrot = self.params['star'].get_value('diffrot','s')
            Period_eq = diffrot + self.params['star'].request_value('rotperiod','s')
            Omega_eq = 2*pi/Period_eq/Omega_crit
            logger.info('rotation frequency (eq) = %.6f Omega_crit'%(Omega_eq))
            surface = 'DiffRotateRoche'
            #-- convert the rotation period values to the coefficients needed
            #   by the marching method
            b1 = Omega*0.54433105395181736
            b2 = roche.diffrotlaw_to_internal(Omega,Omega_eq)
            b3 = 0.
            #-- sanity check: remove these statements when tested enough.
            r0 = -marching.projectOntoPotential(np.array((-0.02, 0.0, 0.0)), surface, b1,b2,b3,1.0).r[0]
            assert(np.allclose(Omega_eq,(b1+b2*r0**2)/0.54433105395181736))
            self.subdivision['mesh_args'] = surface,b1,b2,b3,1.0,r_pole_sol
        elif Omega>=1:
            raise ValueError("Star goes boom! (due to rotation rate being over the critical one [{:.3f}%]".format(Omega*100.))
        
        gridstyle = self.params['mesh'].context
        max_triangles = np.inf # not all algorithms have a limit
        if gridstyle=='mesh:marching':
            #-- marching method. Remember the arguments so that we can reproject
            #   subdivided triangles later on.
            delta = self.params['mesh'].request_value('delta')
            max_triangles = self.params['mesh'].request_value('maxpoints')
            algorithm = self.params['mesh'].request_value('alg')
            if algorithm=='python':
                try:
                    the_grid = marching.discretize(delta,max_triangles,*self.subdivision['mesh_args'][:-1])
                except ValueError:
                    self.save('beforecrash.phoebe')
                    raise
            elif algorithm=='c':
                the_grid = marching.cdiscretize(delta,max_triangles,*self.subdivision['mesh_args'][:-1])
        elif gridstyle=='mesh:wd':
            #-- WD style.
            N = self.params['mesh'].request_value('gridsize')
            the_grid = marching.discretize_wd_style(N,surface,Omega,1.0)
        else:
            raise ValueError("Unknown gridstyle '{}'".format(gridstyle))
        #-- wrap everything up in one array, but first see how many lds columns
        #   we need: for sure the bolometric one, but for the rest, this is
        #   dependent on the pbdep parameters (note that at this point, we just
        #   prepare the array, we don't do anything with it yet):
        N = len(the_grid)
        if N>=(max_triangles-1):
            raise ValueError(("Maximum number of triangles reached ({}). "
                              "Consider raising the value of the parameter "
                              "'maxpoints' in the mesh ParameterSet, or "
                              "decrease the mesh density. It is also "
                              "possible that the equipotential surface is "
                              "not closed.").format(N))
        
        if not 'logg' in self.mesh.dtype.names:
            lds = [('ld___bol','f8',(Nld_law,)),('proj___bol','f8')]
            for pbdeptype in self.params['pbdep']:
                for ipbdep in self.params['pbdep'][pbdeptype]:
                    ipbdep = self.params['pbdep'][pbdeptype][ipbdep]
                    lds.append(('ld_{0}'.format(ipbdep['ref']),'f8',(Nld_law,)))
                    lds.append(('proj_{0}'.format(ipbdep['ref']),'f8'))
                    #lds.append(('velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
                    #ds.append(('_o_velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
            extra = [('logg','f8'),('teff','f8'),('abun','f8')]
            if 'magnetic_field' in self.params:
                extra += [('_o_B_','f8',(3,)),('B_','f8',(3,))]
                logger.info('Added magnetic field columns to mesh')
            dtypes = np.dtype(self.mesh.dtype.descr + extra + lds)
        else:
            dtypes = self.mesh.dtype
        logger.info("covered surface with %d triangles"%(N))
        
        self.mesh = np.zeros(N,dtype=dtypes)
        self.mesh['_o_center'] = the_grid[:,0:3]*r_pole_sol
        self.mesh['center'] = the_grid[:,0:3]*r_pole_sol
        self.mesh['_o_size'] = the_grid[:,3]*r_pole_sol**2
        self.mesh['size'] = the_grid[:,3]*r_pole_sol**2
        self.mesh['_o_triangle'] = the_grid[:,4:13]*r_pole_sol
        self.mesh['triangle'] = the_grid[:,4:13]*r_pole_sol
        self.mesh['_o_normal_'] = the_grid[:,13:16]
        self.mesh['normal_'] = the_grid[:,13:16]
        self.mesh['visible'] = True
        
    
    def update_mesh(self,subset):
        """
        Update the mesh for a subset of triangles
        """
        #-- cut out the part that needs to be updated
        #logger.info('updating %d/%d triangles in mesh'%(sum(subset),len(self.mesh)))
        logger.info('updating triangles in mesh')
        old_mesh = self.mesh[subset].copy()
        #-- remember which arguments were used to create the original mesh
        mesh_args = self.subdivision['mesh_args']
        mesh_args,r_pole_sol = mesh_args[:-1],mesh_args[-1]
        #-- then reproject the old coordinates. We assume they are fairly
        #   close to the real values.
        for tri in range(len(old_mesh)):
            p0 = marching.projectOntoPotential(old_mesh['_o_center'][tri],*mesh_args)
            t1 = marching.projectOntoPotential(old_mesh['_o_triangle'][tri][0:3],*mesh_args)
            t2 = marching.projectOntoPotential(old_mesh['_o_triangle'][tri][3:6],*mesh_args)
            t3 = marching.projectOntoPotential(old_mesh['_o_triangle'][tri][6:9],*mesh_args)
            old_mesh['_o_center'][tri] = p0.r*r_pole_sol
            old_mesh['_o_normal_'][tri] = p0.n
            old_mesh['_o_triangle'][tri][0:3] = t1.r*r_pole_sol
            old_mesh['_o_triangle'][tri][3:6] = t2.r*r_pole_sol
            old_mesh['_o_triangle'][tri][6:9] = t3.r*r_pole_sol
        old_mesh['center'][tri] =        old_mesh['_o_center'][tri]
        old_mesh['normal_'][tri] =       old_mesh['_o_normal_'][tri]
        old_mesh['triangle'][tri][0:3] = old_mesh['_o_triangle'][tri][0:3]
        old_mesh['triangle'][tri][3:6] = old_mesh['_o_triangle'][tri][3:6]
        old_mesh['triangle'][tri][6:9] = old_mesh['_o_triangle'][tri][6:9]
        #-- insert the updated values in the original mesh
        self.mesh[subset] = old_mesh

    def set_time(self,time,ref='all', beaming_alg='none'):
        """
        Set the time of the Star object.
        
        @param time: time
        @type time: float
        @param label: select columns to fill (i.e. bolometric, lcs)
        @type label: str
        """
        logger.info('===== SET TIME TO %.3f ====='%(time))
        # Convert the barycentric time to propertime
        time = self.get_proper_time(time)
        #-- first execute any external constraints:
        self.preprocess(time)
        #-- this mesh is mostly independent of time! We collect some values
        #   that could be handy later on: inclination and rotation frequency
        rotperiod = self.params['star'].request_value('rotperiod','d')
        t0 = self.params['star']['t0'] if 't0' in self.params['star'] else 0.0        
        Omega_rot = 2*pi*(time-t0)/rotperiod
        inclin = self.params['star'].request_value('incl','rad')
        longit = self.params['star'].request_value('long','rad')
        
        #-- check if this Star has spots or is pulsating
        has_spot = 'circ_spot' in self.params
        has_freq = 'puls' in self.params
        
        has_magnetic_field = 'magnetic_field' in self.params
        #-- if time is not set, compute the mesh
        if self.time is None:
            self.compute_mesh(time)
        #-- else, reset to the original values
        elif has_freq:# or has_spot:
            self.reset_mesh()
        
        #-- only compute the velocity if there are spots, pulsations or it was
        #   not computed before
        if self.time is None or has_freq or has_spot:
            self.velocity(ref=ref)
            #-- set the abundance
            self.abundance(time)
            #-- compute polar radius and logg, surface gravity and temperature
            self.surface_gravity()
            #-- if there are any spots, this is taken care of in the function
            #   that calculates the temperature
            self.temperature(time)
            # Set the granulation pattern if necessary
            self.add_granulation(time)
            #-- perhaps add pulsations
            if has_freq:
                self.add_pulsations(time)
            if has_magnetic_field:
                self.magnetic_field()
            #-- compute intensity, rotate to the right position and set default
            #   visible/hidden stuff (i.e. assuming convex shape)
            self.rotate_and_translate(incl=inclin,Omega=longit,theta=Omega_rot,incremental=True)
            
            #if has_freq:
            #    logger.warning("Eclipse detection neglects pulsations")
            #    self.detect_eclipse_horizon(eclipse_detection='hierarchical')
            #else:
            self.detect_eclipse_horizon(eclipse_detection='simple')
        elif rotperiod<np.inf:
            self.velocity(ref=ref)
            self.rotate_and_translate(incl=inclin,Omega=longit,theta=Omega_rot,incremental=False)
            self.detect_eclipse_horizon(eclipse_detection='simple')
        
        self.add_systemic_velocity()
        if self.time is None or has_freq or has_spot:
            self.intensity(ref=ref, beaming_alg=beaming_alg)
        
        #-- remember the time... 
        self.time = time
        self.postprocess(time)
    
    
class BinaryRocheStar(PhysicalBody):    
    """
    Body representing a binary Roche surface.
    
    A BinaryRocheStar strictly has only one obligatory ParameterSet: context
    ``component``. It sets the basic properties such as mass, radius (through
    the potential) and temperature. Because potential is used rather than radius,
    there is a second ParameterSet which is necessary to do any computations
    (though not necessary to initialize): context ``orbit``. It is recommended
    to treat the ``orbit`` also as an obligatory ParameterSet, the exception
    being when working with BinaryBags.
    
    Optional parameterSets:
    
        - ``mesh``: determines mesh properties, such as mesh density and algorithm
        - ``reddening``: set interstellar reddening properties (law, extinction)
        - ``puls``: pulsation mode properties (can be a list)
        - ``circ_spot``: spot properties (can be a list)
        - ``magnetic_field``: global magnetic field properties (dipole...)
        - ``velocity_field``: surface velocity field properties (macroturbulence)
        - ``position``: systemic velocity, position, distance...
        
    As for any other Body, you can give also a list of
    
        - ``pbdep``: passband dependables
        - ``obs``: observations
    
    Useful function:
    
    .. autosummary::
    
        get_component
        volume
        get_mass
    """
    
    _params_tree = dict(component=None, orbit=None, mesh=None, reddening=None,
                        puls='list', circ_spot='list', magnetic_field=None,
                        velocity_field='list', position=None)
    
    
    def __init__(self, component=None, orbit=None, mesh=None, reddening=None,
                 puls=None, circ_spot=None, magnetic_field=None, 
                 velocity_field=None, position=None, pbdep=None, obs=None, **kwargs):
        """
        Initialize a BinaryRocheStar.
        
        Only the :ref:`component <parlabel-phoebe-component>` parameterSet is
        obligatory upon initalization, the rest can be added later. Before doing
        any computations, :envvar:`orbit`, :envvar:`mesh` and :envvar:`pbdep`
        need to be set. If you choose not to do this upon initalization, the
        user is responsible for taking care of this.
        
        All other keyword ParameterSets are optional.        
        """
        
        # Initialize the base body.
        super(BinaryRocheStar,self).__init__(dim=3)
        
        # Create default component and orbit if user doesn't want to do it
        if component is None:
            component = parameters.ParameterSet('component')
        
        if orbit is None:
            orbit = parameters.ParameterSet('orbit')
        
        # Perform some checks on "component", "orbit", and "mesh"
        check_input_ps(self, component, ['component'], 1)
        self.params['component'] = component
        
        check_input_ps(self, orbit, ['orbit'], 'orbit')
        self.params['orbit'] = orbit
        
        if mesh is None:
            self.params['mesh'] = parameters.ParameterSet('mesh:marching')
        else:
            check_input_ps(self, mesh, ['mesh:marching', 'mesh:wd'], 'mesh')
            self.params['mesh'] = mesh
        
        # Prepare for the hiearchical dictionaries that hold the pbdep, obs and
        # syn
        self.params['pbdep'] = OrderedDict()
        self.params['obs'] = OrderedDict()
        self.params['syn'] = OrderedDict()
        self.time = None
        
        # label the body
        self.label = self.params['component']['label']
        
        # Add position parameters, but only if given. DO NOT add default ones,
        # that can be confusing
        if position is not None:
            check_input_ps(self, position, ['position'], 'position')
            self.params['position'] = position
        
        # add interstellar reddening (if none is given, set to the default,
        # this means no reddening
        if reddening is not None:
            check_input_ps(self, reddening, ['reddening:interstellar'], 'reddening')
            self.params['reddening'] = reddening
        
        # Add spot parameters when applicable
        if circ_spot is not None:
            if not isinstance(circ_spot, list):
                to_add = [circ_spot]
            else:
                to_add = circ_spot
            for ito_add in to_add:
                check_input_ps(self, ito_add, ['circ_spot'], 'circ_spot', is_list=True)
            self.params['circ_spot'] = to_add
        
        # Add pulsation parameters when applicable
        if puls is not None:
            if not isinstance(puls, list):
                to_add = [puls]
            else:
                to_add = puls
            for ito_add in to_add:
                check_input_ps(self, ito_add, ['puls'], 'puls', is_list=True)
            self.params['puls'] = to_add
        
        # Add magnetic field parameters when applicable
        if magnetic_field is not None:
            check_input_ps(self, magnetic_field,
                          ['magnetic_field:dipole','magnetic_field:quadrupole'],
                           'magnetic_field')
            self.params['magnetic_field'] = magnetic_field
        
        # Add velocity field parameters when applicable
        if velocity_field is not None:
            check_input_ps(self, velocity_field, ['velocity_field:turb'], 'velocity_field')
            self.params['velocity_field'] = velocity_field
        
        # Parse pbdeps and obs
        if pbdep is not None:
            _parse_pbdeps(self,pbdep)
        if obs is not None:
            _parse_obs(self,obs)
                
        # Check if this star is actually a component in the orbit:
        try:
            this_comp = self.get_component()
        except TypeError:
            this_comp = None
            logger.warning("No orbit specified in BinaryRocheStar. Be sure to do it later.")
        except KeyError:
            this_comp = None
            logger.warning("No orbit specified in BinaryRocheStar. Be sure to do it later.")
        if this_comp is None and orbit is not None:
            raise ValueError(("Cannot figure out which component this is: the "
                              "label in 'component' is '{}', but 'orbit' "
                              "mentions '{}' as the primary, and '{}' as the "
                              "secondary. Please set 'c1label' or 'c2label' "
                              "in 'orbit' to match this component's label"
                              ".").format(component['label'], orbit['c1label'],
                              orbit['c2label']))
                
        # add common constraints
        constraints = ['{sma1} = {sma} / (1.0 + 1.0/{q})',
                       '{sma2} = {sma} / (1.0 + {q})',
                       '{totalmass} = 4*pi**2 * {sma}**3 / {period}**2 / constants.GG',
                       '{mass1} = 4*pi**2 * {sma}**3 / {period}**2 / constants.GG / (1.0 + {q})',
                       '{mass2} = 4*pi**2 * {sma}**3 / {period}**2 / constants.GG / (1.0 + 1.0/{q})',
                       '{asini} = {sma} * sin({incl})',
                       '{com} = {q}/(1.0+{q})*{sma}',
                       '{q1} = {q}',
                       '{q2} = 1.0/{q}']
        for constraint in constraints:
            qualifier = constraint.split('{')[1].split('}')[0]
            if self.params['orbit'] is not None and not self.params['orbit'].has_qualifier(qualifier):
                self.params['orbit'].add_constraint(constraint)
        
        # Initialize the mesh: this makes sure that all the necessary columns
        # are available. If you combine Bodies in a BodyBag, you might want to
        # call "fix_mesh" to make sure the mesh columns are uniform across all
        # Bodies.
        init_mesh(self)
        
        # add morphology constrainer to preprocessing
        self.add_preprocess('binary_morphology')
        
        # Check for leftover kwargs and report to the user
        if kwargs:
            raise ValueError("Unused keyword arguments {} upon initialization of BinaryRocheStar".format(kwargs.keys()))
        
        # Generate a comprehensive log message, that explains what has been
        # added:
        msg = "Created BinaryRocheStar {}".format(self.get_label())
        msg_ = []
        if circ_spot is not None:
            msg_.append('{} circular spots'.format(len(self.params['circ_spot'])))
        if puls is not None:
            msg_.append('{} pulsations'.format(len(self.params['puls'])))
        if magnetic_field is not None:
            msg_.append('a magnetic field')
        if 'pbdep' in self.params:
            for type in self.params['pbdep']:
                msg_.append('{} {}'.format(len(self.params['pbdep'][type]),type))
        if 'obs' in self.params:
            for type in self.params['obs']:
                msg_.append('{} {}'.format(len(self.params['obs'][type]),type))
        if len(msg_):
            msg = msg + ': ' + ', '.join(msg_)
        logger.info(msg)
        self._clear_when_reset['counter'] = 0
        
    
    #@classmethod
    #def add_method(cls, func):
    #    return setattr(cls, func.__name__, types.MethodType(func, cls))
    
    
    def set_label(self, label, component=None):
        """
        Set the label of a BinaryRocheStar.
        
        If :envvar:`component` is None (default), the component will be derived
        automatically. This is usually what you want, except in rare cases when
        initializing the BinaryBag, for example. Then component is either 0
        (primary) or 1 (secondary).
        
        @param label: new label of the Body
        @type label: str
        """
        if component is None:
            this_component = self.get_component()
        else:
            this_component = component
        self.params['component']['label'] = label
        self.params['orbit']['c{}label'.format(this_component+1)] = label
    
    def get_label(self):
        """
        Get the label of the Body.
        
        :return: label of the Body
        :rtype: str
        """
        return self.params['component']['label']
    
    def get_component(self):
        """
        Check which component this is.
        
        @return: 0 (primary) or 1 (secondary) or None (fail)
        @rtype: integer/None
        """
        try:
            if self.get_label()==self.params['orbit']['c1label']:
                return 0
            elif self.get_label()==self.params['orbit']['c2label']:
                return 1
            else:
                return None
        except TypeError:
            raise TypeError("No components (c1label,c2label) in the orbit parameter set")
        
    def get_period(self):
        """
        Extract the main period from the orbit.
        
        Overrides default from Body.
        """
        if not self._main_period:
            period = self.params['orbit']['period']
            t0 = self.params['orbit']['t0']
            shift = self.params['orbit']['phshift']
        else:
            period, t0, shift = super(BinaryRocheStar, self).get_period()
        return period, t0, shift
        
        
    def compute_mesh(self,time=None,conserve_volume=True):
        """
        Compute the mesh of a BinaryRocheStar.
        
        The ``conserve_volume`` parameter doesn't really conserve the volume,
        it only computes the volume so that :py:func:`conserve_volume` can
        do its magic.
        """
        #-- 'derivable' orbital information
        component = self.get_component()+1
        e = self.params['orbit'].get_value('ecc')
        a1 = self.params['orbit'].get_constraint('sma1','m')   # semi major axis of primary orbit
        a2 = self.params['orbit'].get_constraint('sma2','m')   # semi major axis of secondary orbit   
        M  = self.params['orbit'].get_constraint('totalmass','Msol')
        M1 = self.params['orbit'].get_constraint('mass1','kg') # primary mass in solar mass
        M2 = self.params['orbit'].get_constraint('mass2','kg') # secondary mass in solar mass
        q = self.params['orbit'].get_value('q')
        a = self.params['orbit'].get_value('sma','m')
        P = self.params['orbit'].get_value('period','s')
        F = self.params['component'].get_value('syncpar')
        Phi = self.params['component'].get_value('pot')
        com = self.params['orbit'].get_constraint('com','au') / a
        pivot = np.array([com,0,0]) # center-of-mass (should be multiplied by a!)
        T0 = self.params['orbit'].get_value('t0')
        scale = self.params['orbit'].get_value('sma','Rsol')
        
        #-- where in the orbit are we? We need everything in cartesian Rsol units
        #-- dimensionless "D" in Roche potential is ratio of real seperation over
        #   semi major axis.
        pos1, pos2, d = get_binary_orbit(self, time)

        #-- marching method
        if component==2:
            q, Phi = roche.change_component(q, Phi)   
            M1, M2 = M2, M1 # we need to switch the masses!
        #-- is this correct to calculate the polar surface gravity??
        #   I would imagine that we only need the omega_rot due to the binary
        #   period, since it is the angular momentum around the COM that is
        #   important: the polar surface gravity of a rotating star is equal
        #   to that of a nonrotating star!
        omega_rot = F * 2*pi/P # rotation frequency
        omega_orb = 2*pi/P
        r_pole = marching.projectOntoPotential((0,0,1e-5),'BinaryRoche',d,q,F,Phi).r
        r_pole_= np.linalg.norm(r_pole)
        r_pole = r_pole_*a
            
        g_pole = roche.binary_surface_gravity(0,0,r_pole,d*a,omega_orb,M1,M2,normalize=True)
        self.params['component'].add_constraint('{{r_pole}} = {0:.16g}'.format(r_pole))
        self.params['component'].add_constraint('{{g_pole}} = {0:.16g}'.format(g_pole))
        self.params['orbit'].add_constraint('{{d}} = {0:.16g}'.format(d*a))
                
        gridstyle = self.params['mesh'].context
        max_triangles = np.inf # not all mesh algorithms have an upper limit
        if gridstyle=='mesh:marching':
            #-- marching method. Remember the arguments so that we can reproject
            #   subidivded triangles later on.
            delta = self.params['mesh'].request_value('delta')*r_pole_
            max_triangles = self.params['mesh'].request_value('maxpoints')
            algorithm = self.params['mesh'].request_value('alg')
            logger.info('marching {0} {1} {2} {3} (scale={4}, delta={5})'.format(d,q,F,Phi,scale,delta))
            
            if algorithm=='python':
                the_grid = marching.discretize(delta,max_triangles,'BinaryRoche',d,q,F,Phi)
            else:
                the_grid = marching.cdiscretize(delta,max_triangles,'BinaryRoche',d,q,F,Phi)
            logger.info("---> {} triangles".format(len(the_grid)))
        elif gridstyle=='mesh:wd':
            #-- WD style.
            N = self.params['mesh'].request_value('gridsize')
            logger.info('WD grid {0} {1} {2} {3} (scale={4})'.format(d,q,F,Phi,scale))
            the_grid = marching.discretize_wd_style(N,'BinaryRoche',d,q,F,Phi)
        self.subdivision['mesh_args'] = ('BinaryRoche',d,q,F,Phi,scale)
        
        #-- wrap everything up in one array
        N = len(the_grid)
        if N>=(max_triangles-1):
            raise ValueError(("Maximum number of triangles reached ({}). "
                              "Consider raising the value of the parameter "
                              "'maxpoints' in the mesh ParameterSet, or "
                              "decrease the mesh density. It is also "
                              "possible that the equipotential surface is "
                              "not closed.").format(N))
        ld_law = 5
        ldbol_law = 5
        new_dtypes = []
        old_dtypes = self.mesh.dtype.names
        #-- check if the following required labels are in the mesh, if they
        #   are not, we'll have to add them
        required = [('ld___bol','f8',(Nld_law,)),('proj___bol','f8'),
                    ('logg','f8'),('teff','f8'),('abun','f8')]
        for req in required:
            if not req[0] in old_dtypes:
                new_dtypes.append(req)
        if 'pbdep' in self.params:
            for pbdeptype in self.params['pbdep']:
                for ipbdep in self.params['pbdep'][pbdeptype]:
                    ipbdep = self.params['pbdep'][pbdeptype][ipbdep]
                    if not 'ld_{0}'.format(ipbdep['ref']) in old_dtypes:
                        new_dtypes.append(('ld_{0}'.format(ipbdep['ref']),'f8',(Nld_law,)))
                        new_dtypes.append(('proj_{0}'.format(ipbdep['ref']),'f8'))
                        #new_dtypes.append(('velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
                        #new_dtypes.append(('_o_velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
        if new_dtypes:    
            dtypes = np.dtype(self.mesh.dtype.descr + new_dtypes)
        else:
            dtypes = self.mesh.dtype
        #-- the mesh is calculated in units of sma. We need Rsol
        self.mesh = np.zeros(N,dtype=dtypes)
        self.mesh['_o_center'] = the_grid[:,0:3]*scale
        self.mesh['center'] = the_grid[:,0:3]*scale
        self.mesh['_o_size'] = the_grid[:,3]*scale**2
        self.mesh['size'] = the_grid[:,3]*scale**2
        self.mesh['_o_triangle'] = the_grid[:,4:13]*scale
        self.mesh['triangle'] = the_grid[:,4:13]*scale
        self.mesh['_o_normal_'] = -the_grid[:,13:16]
        self.mesh['normal_'] = -the_grid[:,13:16]
        self.mesh['visible'] = True
        
        #-- volume calculations: conserve volume if it is already calculated
        #   before, and of course if volume needs to be conserved.
        #   This is not correct: we need to adapt Omega, not just scale the
        #   mesh. See Wilson 1979
        if conserve_volume:
            if not self.params['component'].has_qualifier('volume'):
                self.params['component'].add_constraint('{{volume}} = {0:.16g}'.format(self.volume()))
                logger.info("volume needs to be conserved at {0}".format(self.params['component'].request_value('volume')))
        
        
    def conserve_volume(self,time,max_iter=10,tol=1e-10):
        """
        Update the mesh to conserve volume.
        
        The value of the potential at which volume is conserved is computed
        iteratively. In the first step, we assume the shape of the star is
        spherical, and compute the volume-change rate via the derivative of the
        Roche potential at the pole. In the next steps, numerical derivatives
        are used.
        
        If ``max_iter==1``, we only reproject the surface, but do not compute
        a new potential value (and thus do not conserve volume, only potential)
        
        @param time: time at which to change the volume
        @type time: float
        @param max_iter: maximum number of iterations
        @type max_iter: int
        @param tol: tolerance cutoff
        @type tol: float
        """
        #-- necessary values
        R = self.params['component'].get_constraint('r_pole') * constants.Rsol
        sma = self.params['orbit']['sma']
        e = self.params['orbit']['ecc']
        P = self.params['orbit']['period'] * 86400.0  # day to second
        q = self.params['orbit']['q']
        F = self.params['component']['syncpar']
        oldpot = self.params['component']['pot']
        M1 = 4*pi**2 * (sma*constants.Rsol)**3 / P**2 / constants.GG / (1.0 + q)
        M2 = q*M1
        component = self.get_component()+1
        
        #-- possibly we need to conserve the volume of the secondary component
        if component == 2:
            q, oldpot = roche.change_component(q, oldpot)  
            M1, M2 = M2, M1
        
        pos1,pos2,d = get_binary_orbit(self, time)
        d_ = d*sma
        omega_rot = F * 2*pi/P # rotation frequency
        omega_orb = 2*pi/P
        
        #-- keep track of the potential vs volume function to compute
        #   derivatives numerically
        potentials = []
        volumes = []
        
        #-- critical potential cannot be subceeded
        if F == 1.:
            critpot = roche.calculate_critical_potentials(q, F, d)[0]
        else:
            critpot = 0.
        
        if max_iter == 0:
            return None
        elif max_iter > 1:
            V1 = self.params['component'].get_constraint('volume')
        else:
            V1 = 1.
        
        for n_iter in range(max_iter):
            #-- compute polar radius
            #R = marching.projectOntoPotential((0,0,1e-5),'BinaryRoche',d,q,F,oldpot).r
            r = [0,0,1e-5]
            R = marching.creproject(np.hstack([r,[0],4*r]).reshape((1,-1)),'BinaryRoche',d,q,F,oldpot)[0,0:3]
            R = np.linalg.norm(R)*sma
            
            #-- these are the arguments to compute the new mesh:
            if oldpot<critpot:
                logger.warning('subceeded critical potential, %.6f--->%.6f'%(oldpot,1.01*critpot))
            #    oldpot = max(oldpot,1.01*critpot)
            self.subdivision['mesh_args'] = ('BinaryRoche',d,q,F,oldpot,sma)
            
            #-- and update the mesh
            self.update_mesh()
            
            #-- compute the volume and keep track of it
            V2 = self.volume()
            potentials.append(oldpot)
            volumes.append(V2)
            
            #-- if we conserved the volume, we're done
            if abs((V2-V1)/V1)<tol:
                break
            
            
            #-- in the first step, we can only estimate the gradient dPot/dV
            if len(potentials)<2:
                oldpot = roche.improve_potential(V2,V1,oldpot,R,d_,q,F,sma=sma)
            #   else, we can approximate in numerically (linear)
            else:
                grad = (potentials[-1]-potentials[-2])/(volumes[-1]-volumes[-2])
                oldpot = grad*(V1-volumes[-2])+potentials[-2]
        else:
            raise ValueError(("Reprojection algorithm failed to converge. If this "
                              "happens at the first computations, probably the "
                              "system is in overcontact status, so check the values "
                              "of the potentials. Otherwise, this is probably "
                              "due to the inherently bad algorithm, which "
                              "builds up errors each time a reprojection is done."
                              " Possible solution is to phase the data, or replace "
                              "the algorithm."))
        
        #-- keep parameters up-to-date
        g_pole = roche.binary_surface_gravity(0,0,R*constants.Rsol,d_*constants.Rsol,omega_orb,M1,M2,normalize=True)
        self.params['component'].add_constraint('{{r_pole}} = {0:.16g}'.format(R*constants.Rsol),do_run_constraints=False)
        self.params['component'].add_constraint('{{g_pole}} = {0:.16g}'.format(g_pole),do_run_constraints=False)
        self.params['orbit'].add_constraint('{{d}} = {0:.16g}'.format(d_*constants.Rsol))
        
        #-- perhaps this was the secondary
        if component==2:
            q,oldpot = roche.change_component(q,oldpot)
        if max_iter>1:
            logger.info("volume conservation (V=%.6f<-->Vref=%.6f): changed potential Pot_ref=%.6f-->Pot_new%.6f)"%(V2,V1,self.params['component']['pot'],oldpot))
            #-- remember the new potential value
            if not 'pot_instant' in self.params['component']:
                self.params['component'].add(dict(qualifier='pot_instant',value=oldpot, cast_type=float, description='Instantaneous potential'))
            else:
                # Avoid having to run the constraints:
                self.params['component'].container['pot_instant'].set_value(oldpot)
                #self.params['component']['pot_instant'] = oldpot
        else:
            logger.info("no volume conservation, reprojected onto instantaneous potential")
        
    def volume(self):
        """
        Compute volume of a BinaryRocheStar.
        """
        norm = coordinates.norm(self.mesh['_o_center'],axis=1)
        return np.sum(self.mesh['_o_size']*norm/3.)
        
    
    @decorators.parse_ref
    def intensity(self,ref='all', beaming_alg='none'):
        """
        Calculate local intensity and limb darkening coefficients.
        """
        parset_isr = dict()
        #-- now run over all labels and compute the intensities
        for iref in ref:
            parset_pbdep,ref = self.get_parset(ref=iref,type='pbdep')
            limbdark.local_intensity(self,parset_pbdep,parset_isr, beaming_alg=beaming_alg)
            
        
        
    def surface_gravity(self):
        """
        Calculate local surface gravity
        """
        
        # Get some info
        component = self.get_component()+1
        q = self.params['orbit']['q']
        if component == 2:
            q = 1.0/q
        a  = self.params['orbit']['sma'] * rsol_2_au # semi-major axis in AU
        asol = self.params['orbit']['sma']
        d  = self.params['orbit'].get_constraint('d') * m_2_au / a
        rp = self.params['component'].get_constraint('r_pole') * m_2_au / a
        gp = self.params['component'].get_constraint('g_pole')
        F  = self.params['component']['syncpar']
        
        # Compute gradients and local gravity
        dOmega_x, dOmega_y, dOmega_z = roche.binary_potential_gradient(self.mesh['_o_center'][:,0]/asol,
                       self.mesh['_o_center'][:,1]/asol,
                       self.mesh['_o_center'][:,2]/asol,
                       q, d, F, normalize=False,
                       output_type='list') # component is not necessary as q is already from component
        
        Gamma_pole = roche.binary_potential_gradient(0, 0, rp, q, d, F,
                                                               normalize=True)
        zeta = gp / Gamma_pole
        grav_local2 = (dOmega_x*zeta)**2 + (dOmega_y*zeta)**2 + (dOmega_z*zeta)**2
        self.mesh['logg'] = 0.5*np.log10(grav_local2) + 2.0
        
        logger.info("derived surface gravity: %.3f <= log g<= %.3f (g_p=%s and Rp=%s Rsol)"%(self.mesh['logg'].min(),self.mesh['logg'].max(),gp,rp*asol))


    def temperature(self,time=None):
        r"""
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
        gravblaw = self.params['component'].get('gravblaw', 'zeipel')
        
        if gravblaw == 'espinosa':
            q = self.params['orbit']['q']
            F = self.params['component']['syncpar']
            sma = self.params['orbit']['sma']
            # To compute the filling factor, we're gonna cheat a little bit: we
            # should compute the ratio of the tip radius to the first Lagrangian
            # point. However, L1 might be poorly defined for weird geometries
            # so we approximate it as 1.5 times the polar radius.
            rp = self.params['component'].request_value('r_pole','Rsol')
            maxr = coordinates.norm(self.mesh['_o_center'],axis=1).max()
            L1 = roche.exact_lagrangian_points(q, F=F, d=1.0, sma=sma)[0]
            rho = maxr / L1
            gravb = roche.zeipel_gravb_binary()(np.log10(q), rho)[0][0]
            self.params['component']['gravb'] = gravb
            logger.info("gravb(Espinosa): F = {}, q = {}, filling factor = {} --> gravb = {}".format(F, q, rho, gravb))
            if gravb>1.0 or gravb<0:
                raise ValueError('Invalid gravity darkening parameter beta={}'.format(gravb))
        
        elif gravblaw == 'claret':
            teff = np.log10(self.params['component']['teff'])
            logg = np.log10(self.params['component'].request_value('g_pole')*100)
            abun = self.params['component']['abun']
            axv, pix = roche.claret_gravb()
            gravb = interp_nDgrid.interpolate([[teff], [logg], [abun]], axv, pix)[0][0]
            logger.info('gravb(Claret): teff = {:.3f}, logg = {:.6f}, abun = {:.3f} ---> gravb = {:.3f}'.format(10**teff, logg, abun, gravb))            
            self.params['component']['gravb'] = gravb
        
        # In any case call the Zeipel law.
        roche.temperature_zeipel(self)
        
        # Perhaps we want to add spots.
        self.add_spots(time)
    
    def add_spots(self,time):
        """
        Adjust the local properties for the presence of spots.
        
        The number of subdivisions is the maximum number of subdivisions for
        all spots. That is, we do not allow to ubdivide one spot two times
        and another one three times: reason: that would need some more
        implementation time.
        """
        if 'circ_spot' in self.params:
            max_subdiv = max([spot_pars['subdiv_num'] for spot_pars in self.params['circ_spot']])
            for j in range(max_subdiv+1):
                last_iter = (j==max_subdiv)
                for i,spot_pars in enumerate(self.params['circ_spot']):
                    logger.info('Spot {}'.format(i))
                    spots.add_circular_spot(self,time,spot_pars,update_temperature=last_iter)
                if not last_iter:
                    self.subdivide(subtype=2)
    
    def add_pulsations(self,time=None):
        component = self.get_component()
        mass = self.params['orbit']['mass{}'.format(component+1)]
        radius = self.params['component']['r_pole']
        F = self.params['component']['syncpar'] 
        orbperiod = self.params['orbit']['period']
        if F>0:
            rotperiod = orbperiod / F
        else:
            rotperiod = np.inf
        
        loc, velo, euler = keplerorbit.get_binary_orbit(time,
                                   self.params['orbit'],
                                   ('primary' if component==0 else 'secondary'))
        
        # The mesh of a BinaryRocheStar rotates along the orbit, and it is
        # independent of the rotation of the star. Thus, we need to specifically
        # specify in which phase the mesh is. It has an "orbital" phase, and a
        # "rotational" phase. We add 90deg so that it has the same orientation
        # as a single star at phase 0.
        
        # to match coordinate system of Star:
        mesh_phase = 0
        # to correct for orbital phase:
        mesh_phase += euler[0]
        #mesh_phase+= (time % orbperiod)/orbperiod * 2*np.pi
        # to correct for rotational phase:
        mesh_phase-= (time % rotperiod)/rotperiod * 2*np.pi
        mesh_phase += np.pi
        
        pulsations.add_pulsations(self, time=time, mass=mass, radius=radius,
                                  rotperiod=rotperiod, mesh_phase=mesh_phase)
    
    def magnetic_field(self, time=None):
        """
        Calculate the magnetic field.
        
        Problem: when the surface is deformed, I need to know the value of
        the radius at the magnetic pole! Or we could just interpret the
        polar magnetic field as the magnetic field strength in the direction
        of the magnetic axes but at a distance of 1 polar radius....
        """
        # Rotational properties
        component = self.get_component()
        F = self.params['component']['syncpar'] 
        orbperiod = self.params['orbit']['period']
        if F>0:
            rotperiod = orbperiod / F
        else:
            rotperiod = np.inf
        
        loc, velo, euler = keplerorbit.get_binary_orbit(time,
                                   self.params['orbit'],
                                   ('primary' if component==0 else 'secondary'))
        
        # The mesh of a BinaryRocheStar rotates along the orbit, and it is
        # independent of the rotation of the star. Thus, we need to specifically
        # specify in which phase the mesh is. It has an "orbital" phase, and a
        # "rotational" phase. We add 90deg so that it has the same orientation
        # as a single star at phase 0.
        
        # to match coordinate system of Star:
        mesh_phase = 0
        # to correct for orbital phase:
        mesh_phase += euler[0]
        # to correct for rotational phase:
        mesh_phase-= (time % rotperiod)/rotperiod * 2*np.pi
        mesh_phase += np.pi
        
        
        # Figure out if we have a dipole or quadrupole
        parset = self.params['magnetic_field']
        context = parset.get_context()
        topology = context.split(':')[-1]
        
        # Some basic quantities we need regardless of the topology
        Bpolar = parset.get_value('Bpolar')
        R = self.params['component'].request_value('r_pole','Rsol')
        r_ = self.mesh['_o_center']
        
        # Then finally get the field according to its topology
        if topology == 'dipole':
            beta = parset.get_value('beta', 'rad')
            phi0 = parset.get_value('phi0', 'rad') - mesh_phase
            B = magfield.get_dipole(time, r_, R, beta, phi0, Bpolar)
        
        elif topology == 'quadrupole':
            beta1 = parset.get_value('beta1', 'rad')
            phi01 = parset.get_value('phi01', 'rad') - mesh_phase
            beta2 = parset.get_value('beta2', 'rad')
            phi02 = parset.get_value('phi02', 'rad') - mesh_phase
            B = magfield.get_quadrupole(time, r_, R, beta1, phi01, beta2, phi02, Bpolar)
        
        # And add it to the mesh!
        self.mesh['_o_B_'] = B
        self.mesh['B_'] = self.mesh['_o_B_']
        
        logger.info("Added {} magnetic field with Bpolar={}G".format(topology, Bpolar))

    
    def abundance(self, time=None):
        """
        Set the abundance of a BinaryRochestar.
        """
        self.mesh['abun'] = list(self.params.values())[0]['abun']
    
    
    def albedo(self, time=None):
        """
        Set the albedo of a BinaryRochestar.
        """
        return None
    
    
    def get_mass(self):
        """
        Compute the mass from the orbit (sma, period, q)
        returned mass will be in solar units
        
        @param return: mass (Msol)
        @type return: float
        """
        return self.params['orbit'].request_value('mass{}'.format(self.get_component()+1), 'Msol') 
    
    
    def projected_velocity(self,los=[0,0,+1],ref=0,method=None):
        rvdep,ref = self.get_parset(ref=ref,type='pbdep')
        ld_func = rvdep.request_value('ld_func')
        method = 'numerical'
        return limbdark.projected_velocity(self,method=method,ld_func=ld_func,ref=ref)
        
    def projected_intensity(self, los=[0.,0.,+1],
                            ref=0, method='numerical', with_partial_as_half=True,
                            beaming_alg='none'):
        """
        Calculate local intensity of a BinaryRocheStar.
        
        We can speed this up if we compute the local intensity first, keep track of the limb darkening
        coefficients and evaluate for different angles. Then we only have to do a table lookup once.
        
        Beaming correction (this is done in limbdark.local_intensity):
        
            - ``beaming_alg='none'``: no beaming correction (possibly taken into account by local_intensity)
            - ``beaming_alg='full'``: no beaming correction (possibly taken into account by local_intensity)
            - ``beaming_alg='local'``: local beaming correction (no limb darkening correction)
            - ``beaming_alg='simple'``: uniform beaming correction
        
        
        """
        if method!='numerical':
            raise ValueError("Only numerical computation of projected intensity of BinaryRocheStar available")

        idep,ref_ = self.get_parset(ref=ref, type='pbdep')
        
        # Fallback to openbol
        if idep is None:
            #raise ValueError("Pbdep with ref '{}' not found in {}".format(ref,self.get_label()))
            ref_ = '__bol'
            ld_func = self.params.values()[0]['ld_func']
        else:
            ld_func = idep['ld_func']
        
        
        proj_int = generic_projected_intensity(self, method=method,
                             ld_func=ld_func, ref=ref_, los=los, 
                             with_partial_as_half=with_partial_as_half,
                             beaming_alg=beaming_alg)
        
        return proj_int
    
    
    
    def set_time(self, time, ref='all', beaming_alg='none'):
        """
        Set the time of a BinaryRocheStar.
        
        The following steps are done:
        
            1. Compute the mesh if it hasn't been computed before
            2. Conserve volume of the mesh if required
            3. Place the mesh in its appropriate location in the orbit
            4. Compute the local surface gravity and temperature
            5. Compute the local intensities
            6. Perform a simple horizon detection.
        
        This function is optimized for circular orbits, i.e the mesh will
        not be recomputed or adapted. Only steps (3) and (6) are then
        excecuted.
        
        @param time: time to be set (d)
        @type time: float
        @param ref: pbdeps to consider for computation of intensities. If
         set to ``all``, all pbdeps are considered.
        @type ref: string (ref) or list of strings
        """
        logger.info('===== SET TIME TO %.3f ====='%(time))
        # Convert the barycentric time to propertime
        time = self.get_proper_time(time)
                
        #-- rotate in 3D in orbital plane
        #   for zero-eccentricity, we don't have to recompute local quantities, and not
        #   even projected quantities (this should be taken care of on a higher level
        #   because other meshes can be in front of it
        #   For non-zero eccentricity, we need to recalculate the grid and recalculate
        #   local quantities
        e = self.params['orbit'].get_value('ecc')
        sma = self.params['orbit'].get_value('sma')#,'Rsol')
        
        has_freq = 'puls' in self.params
        has_spot = 'circ_spot' in self.params
        has_magnetic_field = 'magnetic_field' in self.params
        
        # Having a frequency means we need to do a lot of stuff like for the
        # eccentric case. Let's fake eccentricity even if it's not
        if has_freq and e == 0:
            self.params['orbit']['ecc'] = 1e-10
            e = 1e-10
        
        
        # Our reprojection algorithm doesn't work very well. For eccentric
        # orbits, we'll reset the mesh every 100 reprojections. We can safely
        # remove this part in the future if the reprojection step is
        # improved
        if e > 0:
            if not 'counter' in self._clear_when_reset:
                self._clear_when_reset['counter'] = 0
            elif self._clear_when_reset['counter'] >= 50:
                logger.info("Forcing resetting of mesh (bad reprojection alg)")
                self.time = None
                self._clear_when_reset['counter'] = 0
            else:
                self._clear_when_reset['counter'] += 1
            
        
        #-- there is a possibility to set to conserve volume or equipot
        #   IF eccentricity is zero, we never need to conserve volume, that
        #   is done automatically
        conserve_phase = self.params['orbit'].get('conserve','periastron')
        conserve_volume = e>0
        if conserve_volume and 'conserve' in self.params['orbit']:
            if self.params['orbit']['conserve']=='equipot':
                conserve_volume = False
        # number of iterations for conservation of volume (max)
        if conserve_volume:
            max_iter_volume = 10
        elif e>0:
            max_iter_volume = 1
        else:
            max_iter_volume = 0
            
        #-- we do not need to calculate bolometric luminosities if we don't include
        #   the reflection effect
        do_reflection = False
        #-- compute new mesh if this is the first time set_time is called, or
        #   if the eccentricity is nonzero
            
        if self.time is None or e>0 or has_freq or has_spot:
            if self.time is None:
                #-- if we need to conserve volume, we need to know at which
                #   time. Then we compute the mesh at that time and remember
                #   the value
                if conserve_volume and e>0:
                    cvol_index = ['periastron', 'sup_conj', 'inf_conj',
                                  'asc_node', 'desc_node'].index(conserve_phase)
                    per0 = self.params['orbit'].request_value('per0')/180.*np.pi
                    P = self.params['orbit']['period']
                    t0 = self.params['orbit']['t0']
                    phshift = self.params['orbit']['phshift']
                    t0type = self.params['orbit']['t0type']
                    crit_times = tools.critical_times(self.params['orbit'])
                    logger.info("t0 = {}, t_conserve = {}, {}".format(t0,
                                        crit_times[cvol_index], conserve_phase))
                    self.compute_mesh(crit_times[cvol_index],
                                      conserve_volume=True)
                    # Calculated the basic properties at this time
                    self.surface_gravity()
                    self.abundance(time)
                    self.temperature(time)
                    self.albedo(time)
                    self.intensity(ref=ref, beaming_alg=beaming_alg)
                    self.projected_intensity(beaming_alg=beaming_alg)
                    self.conserve_volume(time,max_iter=max_iter_volume)
                #-- else we still need to compute the mesh at *this* time!
                else:
                    self.compute_mesh(time,conserve_volume=True)
                                        
            #-- else, we have already computed the mesh once, and all we need
            #   to do is either just reset it, or conserve the volume at this
            #   new time point
            else:
                self.reset_mesh()
                self.conserve_volume(time,max_iter=max_iter_volume)
                
            #-- compute polar radius and logg!
            self.surface_gravity()
            self.abundance(time)
            self.temperature(time)
            self.albedo(time)
            
            if has_freq:
                self.add_pulsations(time=time)
            self.intensity(ref=ref, beaming_alg=beaming_alg)
            
            if has_magnetic_field:
                self.magnetic_field(time)
            
            # Compute projected intensity if not done before, to have the
            # passband luminosity
            if self.time is None:
                self.projected_intensity(beaming_alg=beaming_alg)
            
            #-- once we have the mesh, we need to place it into orbit
            keplerorbit.place_in_binary_orbit(self,time)
            
            if do_reflection:
                self.intensity(ref='__bol', beaming_alg=beaming_alg)
        else:
            self.reset_mesh()
            #-- once we have the mesh, we need to place it into orbit
            keplerorbit.place_in_binary_orbit(self,time)
        self.add_systemic_velocity()
        self.detect_eclipse_horizon(eclipse_detection='simple')
        self.time = time
        


class PulsatingBinaryRocheStar(BinaryRocheStar):
    
    def __init__(self, component, puls=None, **kwargs):
        
        # For the rest, this is a normal BinaryRocheStar    
        super(PulsatingBinaryRocheStar,self).__init__(component, **kwargs)
        
        # Add pulsation parameters when applicable
        if puls is not None:
            if not isinstance(puls, list):
                to_add = [puls]
            else:
                to_add = puls
            self.params['puls'] = to_add
            
    def add_pulsations(self,time=None):
        component = self.get_component()
        mass = self.params['orbit']['mass{}'.format(component+1)]
        radius = self.params['component']['r_pole']
        F = self.params['component']['syncpar'] 
        orbperiod = self.params['orbit']['period']
        if F>0:
            rotperiod = orbperiod / F
        else:
            rotperiod = np.inf
        
        loc, velo, euler = keplerorbit.get_binary_orbit(time,
                                   self.params['orbit'],
                                   ('primary' if component==0 else 'secondary'))
        
        # The mesh of a PulsatingBinaryRocheStar rotates along the orbit, and
        # it is independent of the rotation of the star. Thus, we need to
        # specifically specify in which phase the mesh is. It has an "orbital"
        # phase, and a "rotational" phase. We add 90deg so that it has the
        # same orientation as a single star at phase 0.
        
        # to match coordinate system of Star:
        mesh_phase = 0
        # to correct for orbital phase:
        mesh_phase += euler[0]
        #mesh_phase+= (time % orbperiod)/orbperiod * 2*np.pi
        # to correct for rotational phase:
        mesh_phase-= (time % rotperiod)/rotperiod * 2*np.pi
        
        pulsations.add_pulsations(self, time=time, mass=mass, radius=radius,
                                  rotperiod=rotperiod, mesh_phase=mesh_phase)


class MisalignedBinaryRocheStar(BinaryRocheStar):
    
    def get_phase(self,time=None):
        if time is None:
            time = self.__time
        T0 = self.params['orbit'].get_value('t0')
        theta = self.params['orbit'].get_value('theta','rad')
        phi0 = self.params['orbit'].get_value('phi0','rad')
        P = self.params['orbit'].get_value('period','s')
        Pprec = self.params['orbit'].get_value('precperiod','s')
        phi = phi0 - 2*pi*((time-T0)/(P/3600./24.) - (time-T0)/(Pprec/3600./24.))
        return phi
   
    def get_polar_direction(self,time=None,norm=False):
        if time is None:
            time = self.__time
        phi = self.get_phase(time=time)
        theta = self.params['orbit'].get_value('theta','rad')
        coord = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        if not norm:
            coord = coord*1e-5
        return coord
        
    def surface_gravity(self):
        """
        Calculate local surface gravity
        """
        #-- compute gradients and local gravity
        component = self.get_component()+1
        q  = self.params['orbit'].get_constraint('q%d'%(component))
        a  = self.params['orbit'].get_value('sma','au')
        asol = self.params['orbit'].get_value('sma','Rsol')
        d  = self.params['orbit'].get_constraint('d','au')/a
        rp = self.params['component'].get_constraint('r_pole','au')/a
        gp = self.params['component'].get_constraint('g_pole')
        F  = self.params['component'].get_value('f')
        T0 = self.params['orbit'].get_value('t0')
        #Phi = self.params['component'].get_value('pot')
        Phi = self.params['component'].get('pot_instant', self.params['component']['pot'])
        scale = self.params['orbit'].get_value('sma','Rsol')
        theta = self.params['orbit'].get_value('theta','rad')
        phi = self.get_phase()        
        
        M1 = self.params['orbit'].get_constraint('mass1','kg') # primary mass in solar mass
        M2 = self.params['orbit'].get_constraint('mass2','kg') # secondary mass in solar mass
        P = self.params['orbit'].get_value('period','s')
        omega_rot = F * 2*pi/P # rotation frequency
        
        coord = self.get_polar_direction()
        rp = marching.projectOntoPotential(coord,'MisalignedBinaryRoche',d,q,F,theta,phi,Phi).r
        
        dOmega_ = roche.misaligned_binary_potential_gradient(self.mesh['_o_center'][:,0]/asol,
                                                 self.mesh['_o_center'][:,1]/asol,
                                                 self.mesh['_o_center'][:,2]/asol,
                                                 q,d,F,theta,phi,normalize=False) # component is not necessary as q is already from component
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.subplot(221,aspect='equal')
        #plt.scatter(self.mesh['_o_center'][:,0],self.mesh['_o_center'][:,1],c=dOmega_[0],edgecolors='none')
        #plt.colorbar()
        #plt.subplot(222,aspect='equal')
        #plt.scatter(self.mesh['_o_center'][:,0],self.mesh['_o_center'][:,1],c=dOmega_[1],edgecolors='none')
        #plt.colorbar()
        #plt.subplot(223,aspect='equal')
        #plt.scatter(self.mesh['_o_center'][:,0],self.mesh['_o_center'][:,2],c=dOmega_[2],edgecolors='none')
        #plt.colorbar()
        Gamma_pole = roche.misaligned_binary_potential_gradient(rp[0],rp[1],rp[2],q,d,F,theta,phi,normalize=True)        
        zeta = gp / Gamma_pole
        grav_local_ = dOmega_*zeta
        grav_local = coordinates.norm(grav_local_)
        
        #plt.subplot(224,aspect='equal')
        #plt.scatter(self.mesh['_o_center'][:,0],self.mesh['_o_center'][:,1],c=grav_local,edgecolors='none')
        #plt.colorbar()
        #plt.show()
        
        
        #grav_local = roche.misaligned_binary_surface_gravity(self.mesh['_o_center'][:,0]*constants.Rsol,
                                                               #self.mesh['_o_center'][:,1]*constants.Rsol,
                                                               #self.mesh['_o_center'][:,2]*constants.Rsol,asol*constants.Rsol,
                                                               #omega_rot/F,M1,M2,normalize=True,F=F,Rpole=coord*1e5)
        self.mesh['logg'] = conversions.convert('m/s2','[cm/s2]',grav_local)
        logger.info("derived surface gravity: %.3f <= log g<= %.3f (g_p=%s and Rp=%s Rsol)"%(self.mesh['logg'].min(),self.mesh['logg'].max(),gp,rp*asol))
    
    def compute_mesh(self,time=None,conserve_volume=True):
        """
        Compute the mesh.
        
        The ``conserve_volume`` parameter doesn't really conserve the volume,
        it only computes the volume so that :py:func:`conserve_volume` can
        do its magic.
        """
        #-- 'derivable' orbital information
        component = self.get_component()+1
        e = self.params['orbit'].get_value('ecc')
        a1 = self.params['orbit'].get_constraint('sma1','m')   # semi major axis of primary orbit
        a2 = self.params['orbit'].get_constraint('sma2','m')   # semi major axis of secondary orbit   
        M  = self.params['orbit'].get_constraint('totalmass','Msol')
        M1 = self.params['orbit'].get_constraint('mass1','kg') # primary mass in solar mass
        M2 = self.params['orbit'].get_constraint('mass2','kg') # secondary mass in solar mass
        q = self.params['orbit'].get_value('q')
        a = self.params['orbit'].get_value('sma','m')
        P = self.params['orbit'].get_value('period','s')
        F = self.params['component'].get_value('syncpar')
        Phi = self.params['component'].get('pot_instant', self.params['component']['pot'])
        #Phi = self.params['component'].get_value('pot')
        com = self.params['orbit'].get_constraint('com','au') / a
        pivot = np.array([com,0,0]) # center-of-mass (should be multiplied by a!)
        T0 = self.params['orbit'].get_value('t0')
        scale = self.params['orbit'].get_value('sma','Rsol')
        theta = self.params['orbit'].get_value('theta','rad')
        self.__time = time
        phi = self.get_phase(time)
        
        #-- where in the orbit are we? We need everything in cartesian Rsol units
        #-- dimensionless "D" in Roche potential is ratio of real seperation over
        #   semi major axis.
        pos1,pos2,d = get_binary_orbit(self,time)

        #-- marching method
        if component==2:
            q,Phi = roche.change_component(q,Phi)   
            M1,M2 = M2,M1 # we need to switch the masses!
        #-- is this correct to calculate the polar surface gravity??
        #   I would imagine that we only need the omega_rot due to the binary
        #   period, since it is the angular momentum around the COM that is
        #   important: the polar surface gravity of a rotating star is equal
        #   to that of a nonrotating star!
        omega_rot = F * 2*pi/P # rotation frequency
        #-- compute polar radius by projection on the potenial: we need to
        #   make sure we are projecting in the direction of the pole!
        coord = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])*1e-5
        r_pole__ = marching.projectOntoPotential(coord,'MisalignedBinaryRoche',d,q,F,theta,phi,Phi).r        
        r_pole_= np.linalg.norm(r_pole__)
        r_pole = r_pole_*a
        g_pole = roche.misaligned_binary_surface_gravity(r_pole__[0]*a,r_pole__[1]*a,r_pole__[2]*a,
                                              d*a,omega_rot/F,M1,M2,normalize=True)
        self.params['component'].add_constraint('{{r_pole}} = {0:.16g}'.format(r_pole))
        self.params['component'].add_constraint('{{g_pole}} = {0:.16g}'.format(g_pole))
        self.params['orbit'].add_constraint('{{d}} = {0:.16g}'.format(d*a))
        
        gridstyle = self.params['mesh'].context
        max_triangles = np.inf # not all mesh algorithms have an upper limit
        if gridstyle=='mesh:marching':
            #-- marching method. Remember the arguments so that we can reproject
            #   subidivded triangles later on.
            delta = self.params['mesh'].request_value('delta')*r_pole_
            max_triangles = self.params['mesh'].request_value('maxpoints')
            algorithm = self.params['mesh'].request_value('alg')
            logger.info('marching {0} {1} {2} {3} {4} {5} (scale={6})'.format(d,q,F,Phi,theta,phi,scale))
            
            if algorithm=='python':
                the_grid = marching.discretize(delta,max_triangles,'MisalignedBinaryRoche',d,q,F,theta,phi,Phi)
            else:
                the_grid = marching.cdiscretize(delta,max_triangles,'MisalignedBinaryRoche',d,q,F,theta,phi,Phi)
            logger.info("---> {} triangles".format(len(the_grid)))
        elif gridstyle=='mesh:wd':
            #-- WD style.
            N = self.params['mesh'].request_value('gridsize')
            logger.info('WD grid {0} {1} {2} {3} {4} {5} (scale={6})'.format(d,q,F,Phi,theta,phi,scale))
            the_grid = marching.discretize_wd_style(N,'MisalignedBinaryRoche',d,q,F,theta,phi,Phi)
        self.subdivision['mesh_args'] = ('MisalignedBinaryRoche',d,q,F,theta,phi,Phi,scale)
        
        #-- wrap everything up in one array
        N = len(the_grid)
        if N>=(max_triangles-1):
            raise ValueError(("Maximum number of triangles reached ({}). "
                              "Consider raising the value of the parameter "
                              "'maxpoints' in the mesh ParameterSet, or "
                              "decrease the mesh density. It is also "
                              "possible that the equipotential surface is "
                              "not closed.").format(N))
        
        new_dtypes = []
        old_dtypes = self.mesh.dtype.names
        #-- check if the following required labels are in the mesh, if they
        #   are not, we'll have to add them
        required = [('ld___bol','f8',(Nld_law,)),('proj___bol','f8'),
                    ('logg','f8'),('teff','f8'),('abun','f8')]
        for req in required:
            if not req[0] in old_dtypes:
                new_dtypes.append(req)
        if 'pbdep' in self.params:
            for pbdeptype in self.params['pbdep']:
                for ipbdep in self.params['pbdep'][pbdeptype]:
                    ipbdep = self.params['pbdep'][pbdeptype][ipbdep]
                    if not 'ld_{0}'.format(ipbdep['ref']) in old_dtypes:
                        new_dtypes.append(('ld_{0}'.format(ipbdep['ref']),'f8',(Nld_law,)))
                        new_dtypes.append(('proj_{0}'.format(ipbdep['ref']),'f8'))
                        #new_dtypes.append(('velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
                        #new_dtypes.append(('_o_velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
        if new_dtypes:    
            dtypes = np.dtype(self.mesh.dtype.descr + new_dtypes)
        else:
            dtypes = self.mesh.dtype
        #-- the mesh is calculated in units of sma. We need Rsol
        self.mesh = np.zeros(N,dtype=dtypes)
        self.mesh['_o_center'] = the_grid[:,0:3]*scale
        self.mesh['center'] = the_grid[:,0:3]*scale
        self.mesh['_o_size'] = the_grid[:,3]*scale**2
        self.mesh['size'] = the_grid[:,3]*scale**2
        self.mesh['_o_triangle'] = the_grid[:,4:13]*scale
        self.mesh['triangle'] = the_grid[:,4:13]*scale
        self.mesh['_o_normal_'] = -the_grid[:,13:16]
        self.mesh['normal_'] = -the_grid[:,13:16]
        self.mesh['visible'] = True
        
        
        #-- volume calculations: conserve volume if it is already calculated
        #   before, and of course if volume needs to be conserved.
        #   This is not correct: we need to adapt Omega, not just scale the
        #   mesh. See Wilson 1979
        if conserve_volume:
            if not self.params['component'].has_qualifier('volume'):
                self.params['component'].add_constraint('{{volume}} = {0:.16g}'.format(self.volume()))
                logger.info("volume needs to be conserved {0}".format(self.params['component'].request_value('volume')))
        
        
    def conserve_volume(self, time, max_iter=10, tol=1e-6):
        """
        Update the mesh to conserve volume.
        
        The value of the potential at which volume is conserved is computed
        iteratively. In the first step, we assume the shape of the star is
        spherical, and compute the volume-change rate via the derivative of the
        Roche potential at the pole. In the next steps, numerical derivatives
        are used.
        
        If ``max_iter==1``, we only reproject the surface, but do not compute
        a new potential value (and thus do not conserve volume)
        
        @param time: time at which to change the volume
        @type time: float
        @param max_iter: maximum number of iterations
        @type max_iter: int
        @param tol: tolerance cutoff
        @type tol: float
        """
        self.__time = time
        #-- necessary values
        R = self.params['component'].get_constraint('r_pole') * constants.Rsol
        sma = self.params['orbit']['sma']
        e = self.params['orbit']['ecc']
        P = self.params['orbit']['period'] * 86400.0  # day to second
        q = self.params['orbit']['q']
        F = self.params['component']['syncpar']
        oldpot = self.params['component']['pot']
        M1 = self.params['orbit'].get_constraint('mass1')
        M2 = q*M1
        component = self.get_component()+1
        theta = self.params['orbit'].get_value('theta','rad')
        T0 = self.params['orbit']['t0']
        phi = self.get_phase(time)
        
        #-- possibly we need to conserve the volume of the secondary component
        if component==2:
            q,oldpot = roche.change_component(q,oldpot)  
            M1, M2 = M2, M1
        
        pos1,pos2,d = get_binary_orbit(self,time)
        d_ = d*sma
        omega_rot = F * 2*pi/P # rotation frequency
        
        #-- keep track of the potential vs volume function to compute
        #   derivatives numerically
        potentials = []
        volumes = []
                
        if max_iter>1:
            V1 = self.params['component'].request_value('volume')
        else:
            V1 = 1.
        
        coord = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])*1e-5
        
        for n_iter in range(max_iter):
            #-- compute polar radius
            R_ = marching.projectOntoPotential(coord,'MisalignedBinaryRoche',d,q,F,theta,phi,oldpot).r
            R_ = R_*sma
            R = np.linalg.norm(R_)
            g_pole = roche.misaligned_binary_surface_gravity(R_[0]*constants.Rsol,R_[1]*constants.Rsol,R_[2]*constants.Rsol,d_*constants.Rsol,omega_rot/F,M1,M2,normalize=True)
            self.params['component'].add_constraint('{{r_pole}} = {0:.16g}'.format(R*constants.Rsol))
            self.params['component'].add_constraint('{{g_pole}} = {0:.16g}'.format(g_pole))
            #-- these are the arguments to compute the new mesh:
            self.subdivision['mesh_args'] = ('MisalignedBinaryRoche',d,q,F,theta,phi,oldpot,sma)
            
            #-- and update the mesh
            self.update_mesh()
            self.compute_sizes(prefix='_o_')
            self.compute_sizes(prefix='')
            
            #-- compute the volume and keep track of it
            V2 = self.volume()
            potentials.append(oldpot)
            volumes.append(V2)
            
            #-- if we conserved the volume, we're done
            if abs((V2-V1)/V1)<tol:
                break
            
            #-- in the first step, we can only estimate the gradient dPot/dV
            if len(potentials)<2:
                oldpot = roche.improve_potential(V2,V1,oldpot,R,d_,q,F,sma=sma)
            #   else, we can approximate in numerically (linear)
            else:
                grad = (potentials[-1]-potentials[-2])/(volumes[-1]-volumes[-2])
                oldpot = grad*(V1-volumes[-2])+potentials[-2]
                
        
        #-- perhaps this was the secondary
        if component==2:
            q,oldpot = roche.change_component(q,oldpot)
        if max_iter>1:
            logger.info("volume conservation (V=%.6f<-->Vref=%.6f): changed potential Pot_ref=%.6f-->Pot_new%.6f)"%(V2,V1,self.params['component']['pot'],oldpot))
            #-- remember the new potential value
            if not 'pot_instant' in self.params['component']:
                self.params['component'].add(dict(qualifier='pot_instant',value=oldpot, cast_type=float, description='Instantaneous potential'))
            else:
                self.params['component']['pot_instant'] = oldpot
            #self.params['component']['pot'] = oldpot
        else:
            logger.info("no volume conservation, reprojected onto instantaneous potential")
        
        R_ = marching.projectOntoPotential(coord,'MisalignedBinaryRoche',d,q,F,theta,phi,oldpot).r
        R_ = R_*sma
        R = np.linalg.norm(R)
        x1 = roche.misaligned_binary_surface_gravity(R_[0]*constants.Rsol,R_[1]*constants.Rsol,R_[2]*constants.Rsol,d_*constants.Rsol,omega_rot/F,M1,M2,normalize=None)
        R_ = marching.projectOntoPotential(-coord,'MisalignedBinaryRoche',d,q,F,theta,phi,oldpot).r
        R_ = R_*sma
        R = np.linalg.norm(R)
        x2 = roche.misaligned_binary_surface_gravity(R_[0]*constants.Rsol,R_[1]*constants.Rsol,R_[2]*constants.Rsol,d_*constants.Rsol,omega_rot/F,M1,M2,normalize=None)
        
        return list(x1)+list(x2)
        
    def set_time(self,time,ref='all', beaming_alg='none'):
        """
        Set the time of a BinaryRocheStar.
        
        The following steps are done:
        
            1. Compute the mesh if it hasn't been computed before
            2. Conserve volume of the mesh if required
            3. Place the mesh in its appropriate location in the orbit
            4. Compute the local surface gravity and temperature
            5. Compute the local intensities
            6. Perform a simple horizon detection.
                
        @param time: time to be set (d)
        @type time: float
        @param ref: pbdeps to consider for computation of intensities. If
         set to ``all``, all pbdeps are considered.
        @type ref: string (ref) or list of strings
        """
        logger.info('===== SET TIME TO %.3f ====='%(time))
        #-- rotate in 3D in orbital plane
        #   for zero-eccentricity, we don't have to recompute local quantities, and not
        #   even projected quantities (this should be taken care of on a higher level
        #   because other meshes can be in front of it
        #   For non-zero eccentricity, we need to recalculate the grid and recalculate
        #   local quantities
        sma = self.params['orbit'].get_value('sma','Rsol')
        e = self.params['orbit'].get_value('ecc')
        #-- there is a possibility to set to conserve volume or equipot
        #   IF eccentricity is zero, we never need to conserve volume, that
        #   is done automatically
        conserve_phase = self.params['orbit'].get('conserve','periastron')
        conserve_volume = True
        if conserve_volume and 'conserve' in self.params['orbit']:
            if self.params['orbit']['conserve']=='equipot':
                conserve_volume = False
        max_iter_volume = 10 if conserve_volume else 1 # number of iterations for conservation of volume (max)
        #-- we do not need to calculate bolometric luminosities if we don't include
        #   the reflection effect
        do_reflection = False
        #-- compute new mesh if this is the first time set_time is called, or
        #   if the eccentricity is nonzero
        if self.time is None:
            #-- if we need to conserve volume, we need to know at which
            #   time. Then we compute the mesh at that time and remember
            #   the value
            if conserve_volume:
                cvol_index = ['periastron','sup_conj','inf_conj','asc_node','desc_node'].index(conserve_phase)
                per0 = self.params['orbit'].request_value('per0','rad')
                P = self.params['orbit']['period']
                t0 = self.params['orbit']['t0']
                crit_times = keplerorbit.calculate_critical_phases(per0,e)*P + t0
                self.compute_mesh(crit_times[cvol_index],conserve_volume=True)
            #-- else we still need to compute the mesh at *this* time!
            else:
                self.compute_mesh(time,conserve_volume=True)
            #-- the following function both reprojects the surface to the
            #   value of the instantaneous potential and recomputes the
            #   value of the potential to conserve volume (if max_iter>1)
            self.conserve_volume(time,max_iter=max_iter_volume)
        #-- else, we have already computed the mesh once, and all we need
        #   to do is either just reset it, or conserve the volume at this
        #   new time point
        else:
            self.reset_mesh()
            self.conserve_volume(time,max_iter=max_iter_volume)
        #-- once we have the mesh, we need to place it into orbit
        keplerorbit.place_in_binary_orbit(self,time)
        #-- compute polar radius and logg!
        self.surface_gravity()
        self.temperature()
        self.intensity(ref=ref, beaming_alg=beaming_alg)
        if do_reflection:
            self.intensity(ref='__bol', beaming_alg=beaming_alg)
        self.detect_eclipse_horizon(eclipse_detection='simple')
        self.time = time
    
        

class BinaryStar(Star):
    """
    Star with analytical computation of fluxes in a Binary system.
    
    Make sure to also specify C{radius2} in the body parameters, this is
    needed for the computation of the analytical fluxes.
    
    This class inherites everything from Star, and additionaly accepts a
    binary parameter set. It is thus equal to the Star class, with the
    exception of the computation of analytical fluxes in the case the star
    is occulted by a secondary star.
    """
    def __init__(self,star,orbit=None,mesh=None,**kwargs):
        super(BinaryStar,self).__init__(star,mesh,**kwargs)
        self.params['orbit'] = orbit
    
    def get_component(self):
        """
        Check which component this is.
        
        @return: 0 (primary) or 1 (secondary) or None (fail)
        @rtype: integer/None
        """
        #print "Calling get component from BinaryStar",self.label
        if self.get_label()==self.params['orbit']['c1label']:
            return 0
        elif self.get_label()==self.params['orbit']['c2label']:
            return 1
        else:
            return None
    
    def set_time(self, time, *args, **kwargs):
        self.reset_mesh()
        super(BinaryStar,self).set_time(time, *args,**kwargs)
        #keplerorbit.place_in_binary_orbit(self,*args)
        n_comp = self.get_component()
        if n_comp is None:
            raise ValueError(("Cannot figure out which component this is: Body has "
                              "label '{}' while the orbit claims the primary "
                              "has label '{}' and the secondary '{}'. Check "
                              "the labels carefully!").format(self.get_label(),
                                self.params['orbit']['c1label'],
                                self.params['orbit']['c2label']))
        
        component = ('primary','secondary')[n_comp]
        orbit = self.params['orbit']
        loc, velo, euler = keplerorbit.get_binary_orbit(self.time,orbit, component)
        self.rotate_and_translate(loc=loc,los=(0,0,+1),incremental=True)
        self.remove_systemic_velocity()
        self.mesh['velo___bol_'] = self.mesh['velo___bol_'] + velo
        self.add_systemic_velocity()
        #-- once we have the mesh, we need to place it into orbit
        #keplerorbit.place_in_binary_orbit(self,time)
    
    
    def projected_velocity(self,los=[0,0,+1],ref=0):
        rvdep,ref = self.get_parset(ref=ref,type='pbdep')
        method = rvdep['method']
        ld_func = rvdep['ld_func']
        if method=='numerical':
            proj_velo = super(BinaryStar,self).projected_velocity(los=los,ref=ref)
        #-- analytical computation
        elif method=='analytical':
            pos = get_binary_orbit(self,self.time)[self.get_component()]
            proj_velo = pos[-1]
            proj_velo = conversions.convert('m/s','Rsol/d',proj_velo)
            logger.info('RV of Kepler orbit: %.3f Rsol/d'%(proj_velo))
        return proj_velo
    
    
    def projected_intensity(self,los=[0.,0.,+1],ref=0,method=None,
                            with_partial_as_half=True, beaming_alg='none'):
        """
        Calculate local intensity.
        """
        idep,ref = self.get_parset(ref=ref,type='pbdep')
        if method is None:
            method = 'method' in idep and idep['method'] or 'numerical'
            
        
        if method=='numerical':
            return super(BinaryStar,self).projected_intensity(los=los,ref=ref,with_partial_as_half=with_partial_as_half)
        #-- analytical computation
        elif method=='analytical':
            ld_func = idep['ld_func']
            body = list(self.params.values())[0]
            #-- we should scale the flux with the theoretical total flux:
            total_flux = limbdark.sphere_intensity(body,idep)[1]
            prim_orbit = keplerorbit.get_binary_orbit(self.time,self.params['orbit'],'primary')[0]
            secn_orbit = keplerorbit.get_binary_orbit(self.time,self.params['orbit'],'secondary')[0]
            component = self.get_component()
            pos1 = [prim_orbit,secn_orbit][component]
            pos2 = [prim_orbit,secn_orbit][1-component]
            x1,y1 = pos1[0],pos1[1]
            x2,y2 = pos2[0],pos2[1]
            d = sqrt( (x1-x2)**2 + (y1-y2)**2)
            rstar = body.get_value('radius','Rsol')
            rplan = body.get_constraint('radius2','Rsol')
            z = d/rstar
            p = rplan/rstar
            #-- if this star is actually in front of the component, the flux
            #   should be the total flux:
            if pos1[2]>pos2[2]:
                z,p = 5.,1.
            z = np.array([z])
            #-- assume uniform source and dark component
            if ld_func=='uniform':
                logger.info("projected intensity with analytical uniform LD law")
                proj_intens = (transit.occultuniform(z,p)[0])*total_flux
             #-- assume claret limb darkening and dark component
            elif ld_func=='claret':
                logger.info("projected intensity with analytical Claret LD law")
                cn = self.mesh['ld_'+ref][0,:4]
                
                try:
                    proj_intens = (transit.occultnonlin(z,p,cn)[0])*total_flux
                except ValueError:
                    proj_intens = total_flux
            elif ld_func=='linear':
                logger.info("proj. intensity with analytical linear LD law")
                cn = self.mesh['ld_'+ref][0,0], 0.0
                if cn[0] == 0:
                    logger.info("proj. intensity actually uniform LD law")
                    proj_intens = (1-transit.occultuniform(z,p)[0])*total_flux
                else:
                    proj_intens = transit.occultquad(z,p,cn)[0]*total_flux
            elif ld_func=='quadratic':
                logger.info("proj. intensity with analytical quadratic LD law")
                cn = self.mesh['ld_'+ref][0,:2]
                proj_intens = transit.occultquad(z,p,cn)[0]*total_flux
                
            l3 = idep.get('l3', 0.)
            pblum = idep.get('pblum', -1.0)    
            # Scale the projected intensity with the distance
            globals_parset = self.get_globals()
            if globals_parset is not None:
                distance = globals_parset.request_value('distance', 'Rsol')
                proj_intens /= distance**2
            # Take passband luminosity into account
            if pblum >= 0:
                return proj_intens*pblum + l3
            else:
                return proj_intens + l3
            


def serialize(body, description=True, color=True, only_adjust=False,
              filename=None, skip_data=True):
    """
    Attempt no. 1 to serialize a Body.
    """
    def isdata(par):
        value = par.get_value()
        isarray = hasattr(value,'dtype')
        islist = isinstance(value,list)
        if islist and len(value):
            isarray = hasattr(value[0],'dtype')
        if islist and len(value)>10:
            isarray = True
        return isarray
    
    
    #-- if need write to a file, override the defaults
    if filename is not None:
        color = False
        #only_adjust = False
        
    def par_to_str(val,color=True):
        adjust = val.get_adjust()
        # If the parameter is adjustable and has been adjusted (i.e. it has been
        # fitted, ask for it's location and scale parameters).
        if adjust is True and val.has_posterior():
            posterior = val.get_posterior()
            loc, scale = posterior.get_loc(), posterior.get_scale()
            if np.isnan(scale):
                decimals = 6
            else:
                decimals = abs(min(0,int('{:.3e}'.format(scale).split('e')[-1])-3))
            par_repr = '{{:.{:d}g}} +/- {{:.{:d}g}}'.format(decimals, decimals)
            par_repr = par_repr.format(loc, scale)
        else:   
            par_repr = val.to_str()
            
        N = len(par_repr)
        if adjust is True and color is True:
            par_repr = "\033[32m" + '\033[1m' + par_repr + '\033[m'
        elif adjust is False and color is True:
            par_repr = "\033[31m" + '\033[1m' + par_repr + '\033[m'
        elif adjust is True and color is False:
            par_repr = ('*' if not only_adjust else '')+par_repr
            N += 1
        if adjust is True and val.is_lim():
            par_repr = "\033[4m" + par_repr + '\033[m'
        if not adjust and only_adjust:
            return '',0
        return par_repr,N
    
    lines = []
    
    levels = []
    parname_length = 0
    parval_length = 0
    for path,val in body.walk_all():
        if isinstance(val,parameters.Parameter):
            
            if skip_data and isdata(val):
                continue
            
            while len(levels)<(len(path)-1):
                levels.append(0)
            for i,level in enumerate(path[:-1]):
                levels[i] = max(levels[i],len(level))
            parname_length = max(parname_length,len(path[-1]))
            parval_length = max(par_to_str(val,color=False)[1],parval_length)
    levels.append(parname_length)
    for path,val in body.walk_all():
        path = list(path)
        if isinstance(val,parameters.Parameter):
            
            if skip_data and isdata(val):
                continue
            
            par_str,N = par_to_str(val,color=color)
            if N==0:
                continue
            init_length = len(path)
            template = "|".join(['{{:{:d}s}}'.format(level) for level in levels])
            while len(path)<len(levels):
                path.append('')
            if init_length<len(path):
                path[-1] = path[init_length-1]
                path[init_length-1] = ''
            
            
            if description:
                extra_space = len(par_str)-N if par_str>N else N
                parstring = ' = {{:{:d}s}} # {{}}'.format(parval_length+extra_space)
                if val.has_unit():
                    unit = ' ({})'.format(val.get_unit())
                else:
                    unit = ''
                lines.append(template.format(*path)+parstring.format(par_str,val.get_description())+unit)
            else:
                lines.append(template.format(*path)+' = {}'.format(par_str))
    
    txt = "\n".join(lines)
    if filename is not None:
        with open(filename,'w') as ff:
            ff.write(txt)
    else:   
        return txt

# If this module is run from a terminal as a script, execute the unit tests
if __name__=="__main__":
    import doctest
    doctest.testmod()
