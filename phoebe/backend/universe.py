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

Subsection 1.2. PhysicalBody
----------------------------

A L{PhysicalBody} inherits all capabilities of a L{Body}. It extends these
capabilities by making subdivision available, and implementing functions to
generate light curves (L{PhysicalBody.lc}), radial velocity curves
(L{PhysicalBody.rv}), spectra (L{PhysicalBody.spectrum}) and interferometric
visibilities (L{PhysicalBody.ifm}).

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

Subsection 1.4. Subclassing PhysicalBody
----------------------------------------

Specific meshes, i.e. meshes that can easily be parametrized, can be represented
by a class that subclasses the L{PhysicalBody} class for convenience. These kind
of classes facilitate the computation of the mesh. A simple example is
L{Ellipsoid}, parametrized by the radius of the ellipsoid in the x, y and z
direction. One can also add functionality to these classes, which will be
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
from collections import OrderedDict
# Load 3rd party modules
import numpy as np
from numpy import sin,cos,pi,sqrt,pi
from scipy.integrate import quad
from scipy.optimize import nnls
import scipy
try:
    import pylab as pl
except ImportError:
    pass
if enable_mayavi:
    try:
        from enthought.mayavi import mlab
    except ImportError:
        try:
            from mayavi import mlab
        except ImportError:
            enable_mayavi = False
from phoebe.units import conversions
from phoebe.units import constants
from phoebe.utils import coordinates
from phoebe.utils import utils
try:
    from phoebe.utils import cgeometry
except ImportError:
    pass
from phoebe.algorithms import marching
from phoebe.algorithms import subdivision
from phoebe.algorithms import eclipse
from phoebe.backend import decorators
from phoebe.backend import observatory
from phoebe.backend import processing
from phoebe.parameters import parameters
from phoebe.parameters import datasets
from phoebe.atmospheres import roche
from phoebe.atmospheres import limbdark
from phoebe.atmospheres import spots
from phoebe.atmospheres import pulsations
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


#{ Functions of general interest    
def get_binary_orbit(self, time):
    """
    Get the orbital coordinates of a Body at a certain time.
    
    @param self: a Body
    @type self: Body
    @param time: the time in the orbit
    @type time: float
    """
    # Get some information
    P = self.params['orbit'].get_value('period', 'd')
    e = self.params['orbit'].get_value('ecc')
    a = self.params['orbit'].get_value('sma', 'm')
    q = self.params['orbit'].get_value('q')
    a1 = a / (1 + 1.0/q)
    a2 = a - a1
    inclin = self.params['orbit'].get_value('incl','rad')
    argper = self.params['orbit'].get_value('per0','rad')
    long_an = self.params['orbit'].get_value('long_an','rad')
    T0 = self.params['orbit'].get_value('t0')
    
    # Where in the orbit are we? We need everything in cartesian Rsol units
    loc1, velo1, euler1 = keplerorbit.get_orbit(time*24*3600, P*24*3600, e, a1,
                                      T0*24*3600, per0=argper, long_an=long_an,
                                      incl=inclin, component='primary')
    loc2, velo2, euler2 = keplerorbit.get_orbit(time*24*3600, P*24*3600, e, a2,
                                      T0*24*3600, per0=argper, long_an=long_an,
                                      incl=inclin, component='secondary')
    loc1 = np.array(loc1) / a
    loc2 = np.array(loc2) / a
    d = sqrt( (loc1[0]-loc2[0])**2 + \
              (loc1[1]-loc2[1])**2 + \
              (loc1[2]-loc2[2])**2)
    return list(loc1) + list(velo1), list(loc2) + list(velo1), d
    
    
def luminosity(body, ref='__bol'):
    r"""
    Calculate the total luminosity of any object.
    
    This is a numerical general implementation of the intensity moments for
    spheres.
    
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
    
    @return: luminosity of the object (erg/s)
    @rtype: float
    """
    # Set the intensities if they are not calculated yet
    ld_law = body.params.values()[0]['ld_func']
    ld = body.mesh['ld_' + ref]
    if np.all(ld==0):
        body.intensity(ref=ref)
    
    # Get a reference to the mesh, and get the sizes of the triangles in real
    # units
    mesh = body.mesh
    sizes = mesh['size'] * (100*constants.Rsol)**2
    
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
    return (emer_Ibolmu * sizes).sum()
    
    
def load(filename):
    """
    Load a class from a file.
    
    @return: Body saved in file
    @rtype: Body
    """
    ff = open(filename,'r')
    myclass = pickle.load(ff)
    ff.close()
    return myclass


def keep_only_results(system):
    """
    Remove all information from this Body except for the results
    in the C{params} attribute.
    
    Removes parameterSets and meshes.
    
    It can be handy to remove unnecessary information from a Body before
    passing it around via the MPI protocol.
    """
    if hasattr(system, 'params'):
        for key in system.params:
            if not key == 'syn':
                trash = system.params.pop(key)
                del trash
    if hasattr(system, 'bodies'):
        for body in system.bodies:
            keep_only_results(body)
    system.remove_mesh()
    return system


def merge_results(list_of_bodies):
    """
    Merge results of a list of bodies.
    
    Evidently, the bodies need to be representing the same system.
    
    It is of vital importance that each system in the list has the exact
    same hierarchy of results, as they will be iterated over simultaneously.
    It is not important how many or what the `data' and `pbdep' values are.
    
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
    
    This adds columns to the mesh record array according to the dependables.
    
    @param self: the Physical body to set the mesh of
    @type self: PhysicalBody
    """
    #-- wrap everything up in one array, but first see how many lds columns
    #   we need: for sure the bolometric one, but for the rest, this is
    #   dependent on the pbdep parameters (note that at this point, we just
    #   prepare the array, we don't do anything with it yet):
    N = len(self.mesh)
    ld_law = 5
    ldbol_law = 5
    if not 'logg' in self.mesh.dtype.names:
        lds = [('ld___bol', 'f8', (5,)), ('proj___bol', 'f8')]
        for pbdeptype in self.params['pbdep']:
            for iobs in self.params['pbdep'][pbdeptype]:
                iobs = self.params['pbdep'][pbdeptype][iobs]
                lds.append(('ld_{0}'.format(iobs['ref']), 'f8', (5,)))
                lds.append(('proj_{0}'.format(iobs['ref']), 'f8'))
                lds.append(('velo_{0}_'.format(iobs['ref']), 'f8', (3,)))
                lds.append(('_o_velo_{0}_'.format(iobs['ref']), 'f8', (3,)))
        dtypes = np.dtype(self.mesh.dtype.descr + \
                 [('logg','f8'), ('teff','f8'), ('abun','f8')] + lds)
    else:
        dtypes = self.mesh.dtype
    
    # Magnetic field?
    if 'magnetic_field' in self.params:
        dtypes = np.dtype(dtypes.descr + \
                 [('B_', 'f8', (3,)), ('_o_B_','f8', (3,))])
    
    self.mesh = np.zeros(N, dtype=dtypes)


def check_body(body):
    """
    Check consistency of a Body.
    
    We do the following things:
        - If there is only one pbdep, check if the obs have the same reference
        - Check if type of obs is DataSet, type of pbdep is ParameterSet
        - Check if passbands are present in atmosphere table
    """
    # Run over the following categories, and keep track how many are added
    categories = {'lc':[], 'rv':[], 'sp':[], 'if':[], 'pl':[]}
    
    # Get all the references in the body
    refs = body.get_refs()
    refs = body.get_refs(per_category=True)
    
    message = []
    
    for category in refs.keys():
        
        # Keep track of how many pbdeps and obs there are: if there is only one
        # each and their references don't match, we can fix it
        number_pbdeps = []
        number_obs = []
        
        for ref in refs[category]:
            # For each reference, check if the dep and obs exist. This returns
            # the ParameterSet and the reference. If it returns Nones, it does
            # not exist
            ps1, rf1 = body.get_parset(ref=ref, context=category + 'dep')
            ps2, rf2 = body.get_parset(ref=ref, context=category + 'obs')
            
            if rf1 is None and rf2 is not None:
                message.append("Found obs ({}) with ref '{}' but no pbdep to match it".format(category, ref))
                number_obs.append(ps2)
            elif rf1 is not None and rf2 is None:
                #message.append("Found pbdep ({}) with ref '{}' but no observations to match it (in principle this is OK)".format(category, ref))
                number_pbdeps.append(ps1)
            elif rf1 is not None and rf2 is not None:
                number_pbdeps.append(ps1)
                number_obs.append(ps2)
            else:
                raise ValueError("Shouldn't happen. Contact a developer. If you are one, contact another")
            
        # There is one particular case where we can fix things automatically
        # I think: if there is only one pbdep and one obs, they should fit
        # together!
        # OK, this is a little harder than I thought, since I need to fix both
        # the reference and the key in the ordered dict, but the ordered dict
        # can be tucked away deep inside a nested BodyBag. I can't think of an
        # easy way to change the key...
        #if len(number_pbdeps) == 1 and len(number_obs) == 1:
        #    old_ref = number_obs[0]['ref']
        #    new_ref = number_pbdeps[0]['ref']
        #    message.append("Fixed ref {} of category {}".format(ref, category))
        #d2 = OrderedDict([(new_key, v) if k == old_key else (k, v) for k, v in d.items()])
        
    
            
    print("\n".join(message))




def compute_pblum_or_l3(model, obs, sigma=None, pblum=False, l3=False, type='nnls'):
    """
    Rescale the observations to match a model.
    
    obs = pblum * model + l3
    
    Type:
        - nnls does not allow negative coefficients
        - lstsq does allow negative coefficients
    """
    algorithm = dict(nnls=nnls, lstsq=np.linalg.lstsq)[type]
    
    if sigma is None:
        sigma = np.ones_like(obs)
        
    #   only scaling factor
    if pblum and not l3:
        pblum = np.average(obs / model, weights=1.0/sigma**2)
    
    #   only offset
    elif not pblum and l3:
        l3 = np.average(obs - model, weights=1.0/sigma**2)
    
    #   scaling factor and offset
    elif pblum and l3:
        A = np.column_stack([model.ravel(), np.ones(len(model.ravel()))])
        #pblum,l3 = np.linalg.lstsq(A,obs.ravel())[0]
        #pblum, l3 = nnls(A, obs.ravel())[0]
        pblum, l3 = algorithm(A, obs.ravel())[0]
    
    return pblum, l3
                    
def _parse_pbdeps(body, pbdep):
    """
    Attach passband dependables to a body.
    
    This function takes care of separating different types of dependables,
    and attaching them in the C{params} dictionary, an attribute of a
    L{PhysicalBody}. Observables are for example parameterSets of type C{lcdep},
    C{rvdep} or C{spdep} (non-exhaustive list).
    
    First, this function checks if dependables are actually given. That is, it
    cannot be equal to C{None}.
    
    Next, the function checks whether only one single dependable has been
    given. Since we actually expect a list but know what to do with a single
    dependable too, we simply put the single dependable in a list.
    
    If C{body} has no C{pbdep} entry yet in the C{body.params} dictionary,
    a new (ordered) dictionary will be created.
    
    Finally, it checks what types of dependables are given, and each of them
    will be added to the ordered dictionary of the dependable. The key of each
    dependable is its reference (ref).
    
    For each added pbdep, also a "syn" equivalent will be created for
    convenience. It is possible that it stays empty during the course of the
    computations, that's a problem for other functions.
    
    Working with ordered dictionary separated according to dependable type
    enables us to unambiguously reference a certain dependable set with a
    reference (duh), but also with an index. E.g. the first C{lcdep} set that
    is added is referenced by index number 0. This is handy because if you
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
    if not pbdep:
        raise ValueError(('You need to give at least one ParameterSet'
                          'representing dependables'))
        
    # pbdep need to be a list or a tuple. If not, make it one
    elif not isinstance(pbdep, list) and not isinstance(pbdep, tuple):
        pbdep = [pbdep]
    
    # If 'pbdep' is not in the 'params' dictionary, make an empty one
    if not 'pbdep' in body.params:
        body.params['pbdep'] = OrderedDict()
    
    # If 'obs' is not in the 'params' dictionary, make an empty one
    if not 'obs' in body.params:
        body.params['obs'] = OrderedDict()
    
    # If 'syn' is not in the 'params' dictionary, make an empty one
    if not 'syn' in body.params:
        body.params['syn'] = OrderedDict()
    
    # For all parameterSets in pbdep, add them to body.params['pbdep']. This
    # dictionary is itself a dictionary with keys the different contexts, and
    # each entry in that context (ordered) dictionary, has as key the reference
    # and as value the parameterSet.
    parsed_refs = []
    for parset in pbdep:
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
        
        # If the ref already exist, generate a new one
        if ref in body.params['pbdep'][context]:
            ref = str(uuid.uuid4())
            parset['ref'] = ref
        
        # Add parameterSet to dictionaries
        body.params['pbdep'][context][ref] = parset
        
        # Prepare results if they were not already added by the data parser
        if not ref in body.params['syn'][res_context]:
            body.params['syn'][res_context][ref] = \
                              result_sets[context](context=res_context, ref=ref)
            logger.debug(('Prepared results ParameterSet for context '
                         '{} (ref={})'.format(res_context, ref)))
        
        parsed_refs.append(ref)
    
    # That's it
    return parsed_refs
    
    
def _parse_obs(body, data):
    """
    Attach obs to a body.
    
    For each dataset, we also add a 'syn' thingy.
    
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
    pbdep_refs = body.get_refs(per_category=True, include=('pbdep',))
    
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
        if not ref in pbdep_refs[context]:
            logger.warning(("Adding obs with ref='{}', but no corresponding "
                            "pbdeps found. Attempting fix.").format(ref))
            
            # If we have only one dataset and only one pbdep, we can assume
            # they belong together (right?)
            if len(pbdep_refs[context]) == 1 and len(data) == 1:
                ref = pbdep_refs[context][0]
                parset['ref'] = ref
                logger.info("Fix succeeded: {}obs ref fixed to '{}'".format(context, ref))
            
            else:
                logger.info(("Fix failed, there is no obvious match between "
                            "the pbdeps and obs"))
                # If there is only one pbdep reference, assume the user just forgot
                # about setting it, and correct it. Otherwise, raise a ValueError
                raise ValueError(("Adding {context}obs with ref='{ref}', but no "
                              "corresponding {context}deps found (syn cannot "
                              "be computed). I found the following "
                              "{context}dep refs: {av}").format(context=context,
                               ref=ref, av=", ".join(pbdep_refs[context])))
        
        # If the ref already exist, generate a new one. This should never
        # happen, so send a critical message to the user
        if ref in body.params['obs'][data_context]:
            logger.warning(('Data parsing: ref {} already exists!'
                          'Generating new one...'.format(ref)))
            ref = str(uuid.uuid4())
            parset['ref'] = ref
            
        # Prepare results if they were not already added by the data parser
        if not ref in body.params['syn'][res_context]:
            try:
                body.params['syn'][res_context][ref] = \
                         result_sets[data_context](context=res_context, ref=ref)
            except KeyError:
                raise KeyError("Failed parsing obs {}: perhaps not an obs?".format(ref))
            logger.debug(('Prepared results ParameterSet for context '
                          '{} (ref={})'.format(res_context, ref)))
       
        body.params['obs'][data_context][ref] = parset
        parsed_refs.append(ref)
    
    # That's it
    return parsed_refs

#}

class CallInstruct:
    """
    Pass on calls to other objects.
    
    This is really cool!
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
    
    Example:
    
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
        self.time = None
        self.dim = dim
        self.eclipse_detection = eclipse_detection
        self.index = 0
        self.label = str(id(self))
        
        if data is None:
            N = 0
        else:
            N = len(data)
        ft = 'f8' # float type
        mesh = np.zeros(N, dtype=[('_o_center', ft, (dim, )), ('_o_size', ft),
                                  ('_o_triangle', ft, (3*dim, )),
                                  ('_o_normal_', ft,(dim, )),
                                  ('center', ft, (dim, )), ('size', ft),
                                  ('triangle', ft, (3*dim, )),
                                  ('normal_', ft, (dim, )),
                                  ('_o_velo___bol_', ft, (dim, )),
                                  ('velo___bol_', ft, (dim, )), ('mu', ft),
                                  ('partial', bool), ('hidden',bool),
                                  ('visible', bool)])
        if data is not None: # assume we got a record array:
            #-- only copy basic fields
            init_fields = set(mesh.dtype.names)
            fields_given = set(data.dtype.names)
            fields_given_extra = list(fields_given-init_fields)
            fields_given_basic = fields_given & init_fields
            #-- append extra fields
            if fields_given_extra:
                mesh = pl.mlab.rec_append_fields(mesh,fields_given_extra,\
                                  [data[field] for field in fields_given_extra])
            #-- take care of original values and their mutable counter parts.
            for field in fields_given_basic:
                ofield = '_o_%s'%(field)
                if ofield in fields_given:
                    mesh[ofield] = data[ofield]
                elif ofield in mesh.dtype.names:
                    mesh[ofield] = data[field]
                mesh[field] = data[field]
        self.mesh = mesh
        #-- if no information on visibility of the triangles is given, set them
        #   all the visible
        if data is None or not 'visible' in fields_given:
            self.mesh['visible'] = True
        #-- compute extra information upon request
        if compute_centers:
            self.compute_centers()
        if compute_normals:
            self.compute_normals()
        if compute_sizes:
            self.compute_sizes()
        
        #-- keep track of the current orientation, the original (unsubdivided)
        #   mesh and all the parameters of the object.
        self.orientation = dict(theta=0, incl=0, Omega=0, pivot=(0, 0, 0),
                           los=[0, 0, +1], conv='YXZ', vector=[0, 0, 0])
        self.subdivision = dict(orig=None, mesh_args=None, N=None)
        self.params = OrderedDict()
        self._plot = {'plot3D':dict(rv=(-150, 150))}
        #-- the following list of functions will be executed before and
        #   after a call to set_time
        self._preprocessing = []
        self._postprocessing = []
        #-- Add a dict that we can use to store temporary information
        self._clear_when_reset = dict()
        #self.params['pbdep'] = {}
    
    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __str__(self):
        return self.to_string()
    
    def to_string(self,only_adjustable=False):
        """
        String representation of a Body.
        
        @param only_adjustable: only return the adjustable parameters
        @type only_adjustable: bool
        @return: string representation of the parameterSets
        @rtype: str
        """
        txt = ''
        params = self.params
        for param in params:
            if isinstance(params[param],dict):
                for pbdep in params[param]:
                    if isinstance(params[param][pbdep],list):
                        for i,ipbdep in enumerate(params[param][pbdep]):
                            txt += '============ %s/%s/%d ============\n%s\n'%(param,pbdep,i,ipbdep.to_string(only_adjustable=only_adjustable))
                    elif isinstance(params[param][pbdep],dict):
                        for i,lbl in enumerate(params[param][pbdep]):
                            txt += '============ %s/%s/%d (%s) ============\n%s\n'%(param,pbdep,i,lbl,params[param][pbdep][lbl].to_string(only_adjustable=only_adjustable))
                    else:
                        txt += '============ %s ============\n%s\n'%(param,params[param].to_string(only_adjustable=only_adjustable))
            elif isinstance(params[param],list):
                for i,ipbdep in enumerate(params[param]):
                    txt += '============ %s/%d ============\n%s\n'%(param,i,ipbdep.to_string(only_adjustable=only_adjustable))
            else:
                txt += '============ %s ============\n%s\n'%(param,params[param].to_string(only_adjustable=only_adjustable))
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
        Resets the Body but does not clear the results.
        
        After a reset, calling C{set_time} again guarentees that the mesh
        will be recalculated. This could be useful if you want to change
        some basic parameters of an object and force the recomputation of
        the mesh, without the need for creating a new class instance.
        """
        self.time = None
        self._clear_when_reset = dict()
    
    def reset_and_clear(self):
        self.reset()
        self.clear_synthetic()
    
    def walk(self):
        """
        Return iterable to walk through all the parameters of a (nested) Body.
        
        This will recursively return all Parametersets in the Body.
        
        @return: generator of ParameterSets
        @rtype: generator
        """
        return utils.traverse(self,list_types=(BodyBag,Body,list,tuple),dict_types=(dict,))
    
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
                for key in param.values():
                    yield key
    
    def walk_all(self,path_as_string=True):
        """
        Hierarchically walk down a nested Body(Bag).
        
        We need to:
        
            1. Walk through the BodyBags
            2. Walk through the Bodies
            3. Walk through the ParameterSets.
            
        And remember at what level we are!
        
        Each iteration, this function returns a path and a 
        """
        for val,path in utils.traverse_memory(self,list_types=(Body,list,tuple),
                                     dict_types=(dict,),
                                     parset_types=(parameters.ParameterSet,),
                                     get_label=(Body,),
                                     get_context=(parameters.ParameterSet,),
                                     skip=()):
            #-- first one is always root
            path[0] = str(self.__class__.__name__)
            #-- convert to a string if desirable
            if path_as_string:
                for i in range(len(path)):
                    if isinstance(path[i],parameters.ParameterSet):
                        path[i] = path[i].context
                    elif isinstance(path[i],parameters.Parameter):
                        path[i] = path[i].get_qualifier()
                    elif isinstance(path[i],Body):
                        path[i] = path[i].get_label()
                    elif isinstance(path[i],str):
                        continue
                    else:
                        path[i] = '>>>'
                
            yield path,val
    
    def compute(self,*args,**kwargs):
        """
        Call observatory.compute.
        """
        observatory.compute(self,*args,**kwargs)
    
    def compute_pblum_or_l3(self,link=None):
        """
        Compute and set passband luminosity and third light if required.
        
        We need to be able to compute pblum and l3 for all data individually
        if we wish, but we might also want to force certain data to have the
        same pblum and/or l3 (i.e link them). This could make sense for SED
        fits, where pblum is then interpreted as some kind of scaling factor
        (e.g. the angular diameter). Perhaps there are other applications.
        """
        
        # We need observations of course
        if not 'obs' in self.params:
            logger.info('Cannot compute pblum or l3, no observations defined')
            return None
        
        # We'll collect the complete model first (i.e. all the observations in
        # one big 1D array). We'll keep track of the references, so that we know
        # which points represent a particular observation set. Then afterwards,
        # we compute the pblums for all the linked datasets... or for all of
        # separately if link=None.
        if link is not None:
            complete_refrs = []
            complete_model = []
            complete_obser = []
            complete_sigma = []
            complete_pblum = []
            complete_l3 = []
        
        for idata in self.params['obs'].values():
            for observations in idata.values():
                
                # Ignore disabled datasets
                if not observations.get_enabled():
                    continue
                
                # Get the model corresponding to this observation
                model = self.get_synthetic(category=observations.context[:-3],
                                           ref=observations['ref'],
                                           cumulative=True)
                
                # Make sure to have loaded the observations from a file
                loaded = observations.load(force=False)
                
                # Get the "model" and "observations" and their error.
                if observations.context in ['spobs','plobs']:
                    model = np.array(model['flux'])/np.array(model['continuum'])
                    obser = np.array(observations['flux'])/np.array(observations['continuum'])
                    sigma = np.array(observations['sigma'])
                elif observations.context=='lcobs':
                    model = np.array(model['flux'])
                    obser = np.array(observations['flux'])
                    sigma = np.array(observations['sigma'])
                else:
                    logger.error('PBLUM/L3: skipping {}'.format(observations.context))
                    continue
                
                # Determine pblum and l3 for these data if necessary. The pblum
                # and l3 for the model, independently of the observations,
                # should have been computed before when computing the model.
                # Only fit the pblum and l3 here if these parameters are
                # available in the dataset, and they are set to be adjustable
                do_pblum = False
                do_l3 = False
                
                if 'pblum' in observations and observations.get_adjust('pblum'):
                    do_pblum = True
                
                if 'l3' in observations and observations.get_adjust('l3'):
                    do_l3 = True
                
                # Keep track of linking ====> EXPERIMENTAL <======
                if link is not None:
                    complete_model.append(model)
                    complete_obser.append(obser)
                    complete_sigma.append(sigma)
                    complete_refrs.append([observations['ref']]*len(model))
                    complete_pblum.append([do_pblum]*len(model))
                    complete_l3.append([do_l3]*len(model))
                    continue
                
                # Do the computations
                if do_pblum or do_l3:
                    # We allow for negative coefficients in spectra
                    if observations.context in ['plobs','spobs']:
                        alg = 'lstsq'
                    # But not in other stuff
                    else:
                        alg = 'nnls'
                    pblum,l3 = compute_pblum_or_l3(model, obser, sigma, 
                                   pblum=do_pblum, l3=do_l3, type=alg)
                
                #   perhaps we don't need to fit, but we still need to
                #   take it into account
                if not do_pblum and 'pblum' in observations:
                    pblum = observations['pblum']
                elif not do_pblum:
                    pblum = 1.0
                if not do_l3 and 'l3' in observations:
                    l3 = observations['l3']
                elif not do_l3:
                    l3 = 0.0
                #-- set the values and add them to the posteriors
                if do_pblum:
                    observations['pblum'] = pblum
                if do_l3:
                    observations['l3'] = l3
                if loaded:
                    observations.unload()
                
                msg = '{}: pblum={:.6g} ({}), l3={:.6g} ({})'
                logger.info(msg.format(observations['ref'],pblum,\
                            do_pblum and 'computed' or 'fixed',l3,do_l3 \
                            and 'computed' or 'fixed'))
                        
        if link is not None and link=='all':
            model = np.hstack(complete_model)
            obser = np.hstack(complete_obser)
            sigma = np.hstack(complete_sigma)
            do_pblum = np.all(np.hstack(complete_pblum))
            do_l3 = np.all(np.hstack(complete_l3))
            pblum,l3 = compute_pblum_or_l3(model,obser,sigma,
                                           pblum=do_pblum,l3=do_l3)
            for idata in self.params['obs'].values():
                for observations in idata.values():
                    if not do_pblum and 'pblum' in observations:
                        pblum = observations['pblum']
                    elif not do_pblum:
                        pblum = 1.0
                    if not do_l3 and 'l3' in observations:
                        l3 = observations['l3']
                    elif not do_l3:
                        l3 = 0.0
                    #-- set the values and add them to the posteriors
                    if do_pblum:
                        observations['pblum'] = pblum
                    if do_l3:
                        observations['l3'] = l3
            logger.critical('Linking of subsets not yet implemented')
            raise NotImplementedError("Linking of subsets")
                            
    
    def get_logp(self):
        r"""
        Retrieve probability.
        
        If the datasets have passband luminosities C{pblum} and/or third
        light contributions, they will be fitted if they are adjustable.
        
        Every data set has a statistical weight, which is used to weigh them
        in the computation of the total probability.
        
        If ``statweight==0``, then:
        
        .. math::
        
            p = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2_{yi}}}
            \exp\left(-\frac{(y_i - 
            \mathrm{pblum}\ m_i - l_3)^2}{2\sigma_{yi}^2}\right)
            
            \log p = \sum_{i=1}^N -\frac{1}{2}\log(2\pi) - \log(\sigma_{yi})
            - \frac{(y_i - \mathrm{pblum}\ m_i - l_3)^2}{2\sigma_{yi}^2}
            
        Where :math:`p` gives the expected frequency of getting a value
        in an infinitesimal range around :math:`y_i` per unit :math:`dy`.
        To retrieve the :math:`\chi_2`, one can observe that the above is
        equivalent to
        
        .. math::
        
            \log p = K -\frac{1}{2}\chi^2
            
        or
        
        .. math::
        
            K = \log p + \frac{1}{2}\chi^2
            
            \chi^2 = 2 (K - \log p )
            
        
        If ``statweight>1`` each :math:`\log p` value will actually be the mean
        of all :math:`\log p` within one dataset, weighted with the value of
        ``statweight``. This an ugly hack to make some datasets more or less
        important, but is generally not a good approach because it involves
        a subjective determination of the ``statweight`` parameter.
        
        .. warning::
        
            The :math:`\log p` returned by this function is an
            **expected frequency** and not a true probability. That is, the
            :math:`p` comes from the probability density function, not the
            probability. To get the probability itself, you can use scipy on
            the :math:`\chi^2`:
                
            >>> k = Ndata - Npars
            >>> prob = scipy.stats.distributions.chi2.cdf(chi2,k)
            
            If ``prob`` is close to 1 then your model is implausible, if it is
            close to zero it is very plausible.
        
        .. note:: See also
            
            :py:func:`get_chi2 <PhysicalBody.get_chi2>` to compute the
            :math:`\chi^2` statistic and probability
        
        References: [Hogg2009]_.
        
        @return: log probability, chi square, Ndata
        @rtype: float, float, float
        """
        logp = 0.
        chi2 = 0.
        N = 0.
        for idata in self.params['obs'].values():
            for observations in idata.values():
                
                # Ignore disabled datasets
                if not observations.get_enabled():
                    continue
                
                #-- get the model corresponding to this observation
                modelset = self.get_synthetic(category=observations.context[:-3],
                                              ref=observations['ref'],
                                              cumulative=True)
                #-- make sure to have loaded the observations from a file
                loaded = observations.load(force=False)
                #-- get the "model" and "observations" and their error.
                if observations.context=='spobs' or observations.context=='plobs':
                    model = np.array(modelset['flux'])/np.array(modelset['continuum'])
                    obser = np.array(observations['flux'])/np.array(observations['continuum'])
                    sigma = np.array(observations['sigma'])
                elif observations.context=='lcobs':
                    model = np.array(modelset['flux'])
                    obser = np.array(observations['flux'])
                    sigma = np.array(observations['sigma'])
                elif observations.context=='ifobs':
                    model = np.array(modelset['vis2'])
                    obser = np.array(observations['vis2'])
                    sigma = np.array(observations['sigma_vis2'])
                elif observations.context=='rvobs':
                    model = conversions.convert('Rsol/d','km/s',np.array(modelset['rv']))
                    obser = np.array(observations['rv'])
                    sigma = np.array(observations['sigma'])
                else:
                    raise NotImplementedError('probability for {}'.format(observations.context))                
                #-- take pblum and l3 into account:
                pblum = observations['pblum'] if ('pblum' in observations) else 1.0
                l3 = observations['l3'] if ('l3' in observations) else 0.0
                #-- compute the log probability ---> not sure that I need to do sigma*pblum, I'm not touching the observations!
                term1 = - 0.5*np.log(2*pi*(sigma)**2)
                term2 = - (obser-model*pblum-l3)**2/(2.*(sigma)**2)
                #-- do also the Stokes V profiles. Becuase they contain the
                #   derivative of the intensity profile, the l3 factor disappears
                if observations.context=='plobs':
                    if 'V' in observations['columns']:
                        model = np.array(modelset['V'])/np.array(modelset['continuum'])
                        obser = np.array(observations['V'])/np.array(observations['continuum'])
                        sigma = np.array(observations['sigma_V'])
                        term1 += - 0.5*np.log(2*pi*(sigma)**2)
                        term2 += - (obser-model*pblum)**2/(2.*(sigma)**2)
                
                #-- statistical weight:
                statweight = observations['statweight']
                #   if stat_weight is negative, we try to determine the
                #   effective number of points:
                # ... not implemented yet ...
                #   else, we take take the mean and multiply it with the
                #   weight:
                if statweight>0:
                    this_logp = (term1 + term2).mean()*statweight
                    this_chi2 = -(2*term2).mean()*statweight
                #   if statistical weight is zero, we don't do anything:
                else:
                    this_logp = (term1 + term2).sum()
                    this_chi2 = -2*term2.sum()
                logger.info("Statist weight of {} = {}".format(observations['ref'],statweight))
                logger.info("pblum = {:.3g}, l3 = {:.3g}".format(pblum,l3))
                logger.info("Chi2 of {}".format(observations['ref'],term2*2))
                logp += this_logp
                chi2 += this_chi2
                N += len(obser)
                if loaded:
                    observations.unload()
        return logp, chi2, N
    
    def get_chi2(self):
        r"""
        Return the :math:`\chi^2` and resulting probability of the model.
        
        If ``prob`` is close to unity, the model is implausible, if it is
        close to zero, it is very plausible.
        """
        #-- get the necessary info
        logp, chi2, Ndata = self.get_logp()
        adj = self.get_adjustable_parameters()
        Npar = len(adj)
        #-- compute the chi2 probability
        k = Ndata - Npar
        prob = scipy.stats.distributions.chi2.cdf(chi2,k)
        #-- that's it!
        return chi2, prob, Ndata, Npar
        
    
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
                
                #-- make sure to have loaded the observations from a file
                loaded = observations.load(force=False)
                if observations.context=='spobs':
                    obser_ = np.ravel(observations['flux']/observations['continuum'])
                    sigma_ = np.ravel(observations['sigma'])
                elif observations.context=='lcobs':
                    obser_ = np.ravel(observations['flux'])
                    sigma_ = np.ravel(observations['sigma'])
                elif observations.context=='ifobs':
                    obser_ = np.ravel(observations['vis2'])
                    sigma_ = np.ravel(observations['sigma_vis2'])
                elif observations.context=='rvobs':
                    obser_ = np.ravel(observations['rv'])
                    sigma_ = np.ravel(observations['sigma'])
                else:
                    raise NotImplementedError('probability')  
                #-- append to the "whole" model.
                mu.append(obser_)
                sigma.append(sigma_)
                if loaded:
                    observations.unload()
        return np.hstack(mu),np.hstack(sigma)
    
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
                #-- make sure to have loaded the observations from a file
                loaded = observations.load(force=False)
                if observations.context=='spobs' or observations.context=='plobs':
                    model_ = np.ravel(np.array(modelset['flux'])/np.array(modelset['continuum']))
                    obser_ = np.ravel(np.array(observations['flux'])/np.array(observations['continuum']))
                    sigma_ = np.ravel(np.array(observations['sigma']))
                elif observations.context=='lcobs':
                    model_ = np.ravel(np.array(modelset['flux']))
                    obser_ = np.ravel(np.array(observations['flux']))
                    sigma_ = np.ravel(np.array(observations['sigma']))
                elif observations.context=='ifobs':
                    model_ = np.ravel(np.array(modelset['vis2']))
                    obser_ = np.ravel(np.array(observations['vis2']))
                    sigma_ = np.ravel(np.array(observations['sigma_vis2']))
                elif observations.context=='rvobs':
                    model_ = conversions.convert('Rsol/d','km/s',np.ravel(np.array(modelset['rv'])))
                    obser_ = np.ravel(np.array(observations['rv']))
                    sigma_ = np.ravel(np.array(observations['sigma']))
                else:
                    raise NotImplementedError('probability')
                
                #-- statistical weight:
                statweight = observations['statweight']
                #-- take pblum and l3 into account:
                pblum = observations['pblum'] if ('pblum' in observations) else 1.0
                l3 = observations['l3'] if ('l3' in observations) else 0.0
                model_ = pblum*model_ + l3
                
                if observations.context=='plobs':
                    if 'V' in observations['columns']:
                        # We need to correct the Stokes profile for the passband
                        # luminosity factor, as this was not taken into account
                        # during the calculations
                        model_ = np.hstack([model_,np.ravel(np.array(modelset['V'])/np.array(modelset['continuum'])*pblum)])
                        obser_ = np.hstack([obser_,np.ravel(np.array(observations['V'])/np.array(observations['continuum']))])
                        sigma_ = np.hstack([sigma_,np.ravel(np.array(observations['sigma_V']))])
                
                #-- transform to log if necessary:
                if 'fittransfo' in observations and observations['fittransfo']=='log10':
                    sigma_ = sigma_/(obser_*np.log(10))
                    model_ = np.log10(model_)
                    obser_ = np.log10(obser_)
                    logger.info("Transformed model to log10 for fitting")
                
                #-- append to the "whole" model.
                model.append(model_)
                mu.append(obser_)
                sigma.append(sigma_/statweight**2)
                if loaded:
                    observations.unload()
        if not len(mu):
            mu = np.array([])
            sigma = np.array([])
            model = np.array([])
            return mu,sigma,model
        else:
            return np.hstack(mu),np.hstack(sigma),np.hstack(model)
    
    def get_adjustable_parameters(self):
        mylist = []
        for path,val in self.walk_all():
            path = list(path)
            if isinstance(val,parameters.Parameter) and val.get_adjust():
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
    
    def add_preprocess(self,func,*args,**kwargs):
        """
        Add a preprocess to the Body.
        
        The list of preprocessing functions are executed before set_time is
        called.
        
        @param func: name of a processing function in backend.processes
        @type func: str
        """
        self._preprocessing.append((func,args,kwargs))

    def add_postprocess(self,func,*args,**kwargs):
        """
        Add a postprocess to the Body.
        
        @param func: name of a processing function in backend.processes
        @type func: str
        """
        self._postprocessing.append((func,args,kwargs))
    
        
    def preprocess(self,time):
        """
        Run the preprocessors.
        
        @param time: time to which the Body will be set
        @type time: float or None
        """
        for func,arg,kwargs in self._preprocessing:
            getattr(processing,func)(self,time,*arg,**kwargs)
    
    def postprocess(self,time):
        """
        Run the postprocessors.
        """
        for func,args,kwargs in self._postprocessing:
            getattr(processing,func)(self,time,*args,**kwargs)
    
    def set_values_from_priors(self):
        """
        Set values from adjustable parameters with a prior to a random value
        from it's prior.
        """
        walk = utils.traverse(self,list_types=(BodyBag,Body,list,tuple),dict_types=(dict,))
        for parset in walk:
            #-- for each parameterSet, walk through all the parameters
            for qual in parset:
                #-- extract those which need to be fitted
                if parset.get_adjust(qual) and parset.has_prior(qual):
                    parset.get_parameter(qual).set_value_from_prior()
                
    
    
    #{ Functions to manipulate the mesh    
    def detect_eclipse_horizon(self,eclipse_detection=None,**kwargs):
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
        
        if eclipse_detection=='hierarchical':
            eclipse.detect_eclipse_horizon(self)
        elif eclipse_detection=='simple':
            threshold = kwargs.get('threshold',185./180.*pi)
            partial = np.abs(self.mesh['mu'])>=threshold
            visible = self.mesh['mu']>0 & -partial
            
            self.mesh['visible'] = visible
            self.mesh['hidden'] = -visible & -partial
            self.mesh['partial'] = partial
        else:
            raise ValueError("don't know how to detect eclipses/horizon (set via parameter 'eclipse_detection'")
    
    def rotate_and_translate(self,theta=0,incl=0,Omega=0,
              pivot=(0,0,0),loc=(0,0,0),los=(0,0,+1),incremental=False,
              subset=None):
        #-- select a subset (e.g. of partially visible triangles) or not?
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
        #-- rotate
        mesh = coordinates.rotate_and_translate(mesh,
                       theta=theta,incl=incl,Omega=Omega,
                       pivot=pivot,los=los,loc=loc,incremental=incremental)
        #-- replace the subset or not?
        if subset is not None:
            self.mesh[subset] = mesh
        else:
            self.mesh = mesh
        #-- remember the orientiation, maybe this is useful at some point.
        self.orientation['theta'] = theta
        self.orientation['incl'] = incl
        self.orientation['Omega'] = Omega
        self.orientation['pivot'] = pivot
        self.orientation['los'] = los
        self.orientation['vector'] = loc
    
    
    
    
    
    def compute_centers(self):
        """
        Compute the centers of the triangles.
        """
        for idim in range(self.dim):
            self.mesh['_o_center'][:,idim] = self.mesh['_o_triangle'][:,idim::self.dim].sum(axis=1)/3.
            self.mesh['center'][:,idim] = self.mesh['triangle'][:,idim::self.dim].sum(axis=1)/3.
    
    def compute_sizes(self,prefix=''):
        """
        Compute triangle sizes from the triangle vertices
        """
        #-- size of triangle
        side1 = self.mesh[prefix+'triangle'][:,0*self.dim:1*self.dim]-self.mesh[prefix+'triangle'][:,1*self.dim:2*self.dim]
        side2 = self.mesh[prefix+'triangle'][:,0*self.dim:1*self.dim]-self.mesh[prefix+'triangle'][:,2*self.dim:3*self.dim]
        side3 = self.mesh[prefix+'triangle'][:,1*self.dim:2*self.dim]-self.mesh[prefix+'triangle'][:,2*self.dim:3*self.dim]
        a = sqrt(np.sum(side1**2,axis=1))
        b = sqrt(np.sum(side2**2,axis=1))
        c = sqrt(np.sum(side3**2,axis=1))
        s = 0.5*(a+b+c)
        self.mesh[prefix+'size'] = sqrt( s*(s-a)*(s-b)*(s-c))
        #print 'CSA',self.mesh[prefix+'size'].sum()
        
    def compute_normals(self,prefixes=('','_o_')):
        """
        Compute normals from the triangle vertices.
        
        The normal on a triangle is defined as the cross product of the edge
        vectors::
            
            N = E(0,1) X E(0,2)
        
        Comparison between numeric normals and true normals for a sphere. The
        numeric normals are automatically computed for an Ellipsoid. For a
        sphere, they are easily defined.
        
        >>> sphere = Ellipsoid()
        >>> sphere.compute_mesh(delta=0.3)
        
        >>> p = mlab.figure()
        >>> sphere.plot3D(normals=True)
        
        >>> sphere.mesh['normal_'] = sphere.mesh['center']
        >>> sphere.plot3D(normals=True)
        >>> p = mlab.gcf().scene.camera.parallel_scale=0.3
        >>> p = mlab.draw()
            
        """
        #-- normal is cross product of two sides
        for prefix in prefixes:
            side1 = self.mesh[prefix+'triangle'][:,0:0+self.dim]-self.mesh[prefix+'triangle'][:,1*self.dim:2*self.dim]
            side2 = self.mesh[prefix+'triangle'][:,0:0+self.dim]-self.mesh[prefix+'triangle'][:,2*self.dim:3*self.dim]
            self.mesh[prefix+'normal_'] = np.cross(side1,side2)

    def area(self):
        return self.mesh['size'].sum()
    
    def get_coords(self,type='spherical',loc='center'):
        """
        Return the coordinates of the star in a convenient coordinate system.
        
        Phi is longitude
        theta is colatitude
        
        Can be useful for surface maps or so.
        
        Nees some work...
        """
        index = np.array([1,0,2])
        r1,phi1,theta1 = coordinates.cart2spher_coord(*self.mesh['_o_triangle'][:,0:3].T[index])
        r2,phi2,theta2 = coordinates.cart2spher_coord(*self.mesh['_o_triangle'][:,3:6].T[index])
        r3,phi3,theta3 = coordinates.cart2spher_coord(*self.mesh['_o_triangle'][:,6:9].T[index])
        r4,phi4,theta4 = coordinates.cart2spher_coord(*self.mesh['_o_center'].T[index])
        #r = np.hstack([r1,r2,r3,r4])
        #phi = np.hstack([phi1,phi2,phi3,phi4])
        #theta = np.hstack([theta1,theta2,theta3,theta4])
        if loc=='center':
            return r4,phi4,theta4
        else:
            table = np.column_stack([phi1,theta1,r1,phi2,theta2,r2,phi3,theta3,r3])
            return table
    
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
        
        If possible, please give the context. There is still a bug in the
        code below, that does not iterate over bodies in a lower level if nothing
        was found in this body. Have a look at the first part of the code to
        find the small workaround for that.
        
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
            logger.info("Requested bolometric parameterSet")
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
                elif ref_is_string and ref < n_refs:
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
                    logger.info("Requested parset ref={}, context={} but it does not seem to exist".format(ref,context))
                    return None, None
                
                logger.info("Requested parset ref={}, context={} and found ref={}, context={}".format(ref,context,ps['ref'],ps.get_context()))
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
                logger.info("Requested parset ref={}, context={} but it does not exist".format(ref,context))
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
                    raise ValueError('No {} defined: looking in type={}, available={}'.format(icat,type,list(self.params[type].keys())))
                counter = 0
                for ips in self.params[type][icat]:
                    ps = self.params[type][icat][ips]
                    is_ref = ('ref' in ps) and (ps['ref']==ref)
                    is_number = counter==ref
                    if is_ref or is_number:
                        logger.info("Requested parset ref={}, type={}, category={} and found ref={}, context={}".format(ref,type,category,ps['ref'],ps.get_context()))
                        return ps,ps['ref']
                    counter += 1
            return None,None
    
    def list(self, summary=None):
        """
        List with indices all the parameterSets that are available.
        
        ``summary`` can be None, 'short' or long.
        
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

        """
        def add_summary_short(thing, text):
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
               for category in ['lc', 'rv', 'sp', 'if', 'pl']:
                   lbl = (category+ptype[-3:])
                   if ptype in thing.params and lbl in thing.params[ptype]:
                       this_type.append('{} {}'.format(len(thing.params[ptype][lbl]),lbl))
                       ns += len(thing.params[ptype][lbl])
               
               # Only report if there are some
               if ns > 0:
                   mystring = this_type[0]+', '.join(this_type[1:])
                   summary.append("\n".join(textwrap.wrap(mystring, initial_indent=indent, subsequent_indent=indent+5*' ')))
           
           text += summary
        
        
        def add_summary_long(thing, text):
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
            for category in ['lc', 'rv', 'sp', 'if', 'pl']:
               
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
        
        
        def emphasize(text):
            return '\033[1m\033[4m' + text + '\033[m'    
        
        if summary:
            add_summary = locals()['add_summary_'+summary]
        else:
            add_summary = lambda thing, text: ''
        
        # Top level string: possible the BB has no label
        try:   
            text = ["{} ({})".format(self.get_label(),self.__class__.__name__)]
        except ValueError:
            text = ["<nolabel> ({})".format(self.__class__.__name__)]
        
        text[-1] = emphasize(text[-1])
        
        # Add the summary of the top level thing
        add_summary(self, text)

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
                
            # If this thing is a BodyBag, treat it as such
            if isinstance(thing, BodyBag) and previous_label != label:
                
                # A body bag is represented by a "|   |    +=======>" sign
                level = len(loc)-1 
                prefix = ('|'+(' '*9))*(level-2)
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
            add_summary(thing, text)
            
            # Remember the label
            previous_label = label
            
        print("\n".join(text))
    
    
    def clear_synthetic(self):
        """
        Clear the body from all calculated results.
        """
        result_sets = dict(lcsyn=datasets.LCDataSet,
                       rvsyn=datasets.RVDataSet,
                       spsyn=datasets.SPDataSet,
                       ifsyn=datasets.IFDataSet,
                       plsyn=datasets.PLDataSet)
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
    
    def __add__(self, other):
        """
        Combine two bodies in a BodyBag.
        """
        return BodyBag([self, other])
    #}
    
    
    #{ Input and output
    def plot2D(self,**kwargs):
        """
        Plot mesh in 2D using matplotlib.
        
        For more information, see :py:func:`phoebe.backend.observatory.image`.
        """
        return observatory.image(self,**kwargs)
        
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
            scale_factor = kwargs.pop('scale_factor',1.)
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
    
    
    def save(self,filename):
        """
        Save a class to an file.
        
        You need to purge signals before writing.
        """
        ff = open(filename,'w')
        pickle.dump(self,ff)
        ff.close()  
        logger.info('Saved model to file {} (pickle)'.format(filename))
    
    def copy(self):
        """
        Copy this instance.
        """
        return copy.deepcopy(self)
    
    
    @decorators.parse_ref
    def ifm(self,ref='allifdep',time=None):
        """
        You can only do this if you have observations attached.
        """
        #-- don't bother if we cannot do anything...
        if hasattr(self,'params') and 'obs' in self.params and 'ifobs' in self.params['obs']:
            for lbl in set(ref):
                ifobs,lbl = self.get_parset(type='obs',ref=lbl)
                times = ifobs['time']
                posangle = np.arctan2(ifobs['vcoord'],ifobs['ucoord'])/pi*180.
                baseline = sqrt(ifobs['ucoord']**2 + ifobs['vcoord']**2)
                eff_wave = None if (not 'eff_wave' in ifobs or not len(ifobs['eff_wave'])) else ifobs['eff_wave']
                if time is None:
                    keep = np.ones(len(posangle),bool)
                else:
                    keep = np.abs(times-time)<1e-8
                #-- if nothing needs to be computed, don't do it
                if sum(keep)==0:
                    continue
                output = observatory.ifm(self,posangle=posangle[keep],
                                     baseline=baseline[keep],eff_wave=eff_wave,
                                     ref=lbl,keepfig=False)
                                     #ref=lbl,keepfig=('pionier_time_{:.8f}'.format(time)).replace('.','_'))
                ifsyn,lbl = self.get_parset(type='syn',ref=lbl)
                ifsyn['time'] += [time]*len(output[0])
                ifsyn['ucoord'] += list(ifobs['ucoord'][keep])
                ifsyn['vcoord'] += list(ifobs['vcoord'][keep])
                ifsyn['vis2'] += list(output[3])
                ifsyn['phase'] += list(output[4])
        #-- try to descend into a bodyBag
        else:
            try:
                for body in self.bodies:
                    body.ifm(ref=ref,time=time)
            except AttributeError:
                pass
    
    @decorators.parse_ref
    def pl(self, ref='allpldep', time=None, obs=None):
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
                
            # Expand output and save it to the synthetic thing
            wavelengths_, I, V, Q , U, cont = output
            
            base['time'].append(self.time)
            base['wavelength'].append(wavelengths_ / 10.)
            base['flux'].append(I)
            base['V'].append(V)
            base['Q'].append(Q)
            base['U'].append(U)
            base['continuum'].append(cont)
    
    @decorators.parse_ref
    def sp(self, ref='allspdep', time=None, obs=None):
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
            
            base['time'].append(self.time)
            base['wavelength'].append(wavelengths_ / 10.)
            base['flux'].append(I)
            base['continuum'].append(cont)
    #}
    
    

class PhysicalBody(Body):
    """
    Extends a Body with extra functions relevant to physical bodies.
    
    **Adding/remove parameters and data**
    
    .. autosummary::
    
       add_obs
       add_pbdeps
       remove_dependables
       remove_obs
    
    **Resetting/clearing**
    
    .. autosummary::
        
       Body.clear_synthetic
       Body.reset
       clear_reflection
       prepare_reflection
       remove_mesh
       reset_mesh
       unsubdivide
       
    **Request (additional) information**
    
    .. autosummary::
    
       Body.get_parset
       Body.get_refs
       Body.get_synthetic
       Body.get_coords
       Body.get_label
       Body.list
       get_parameters
       as_point_source
    
    **Statistics**
    
    .. autosummary::
    
       Body.get_logp
       Body.get_chi2
       Body.get_model
    
    **Iterators**
    
    .. autosummary::
    
       Body.walk
       Body.walk_type
    
    **Compute passband dependent quantities**
    
    .. autosummary::
    
        ifm
        lc
        ps
        rv
        sp
        
    **Body computations**
     
    .. autosummary::
       
       Body.compute_centers
       Body.compute_normals
       Body.compute_sizes
       Body.compute_pblum_or_l3
       Body.detect_eclipse_horizon
       subdivide
       update_mesh
    
    **Input/output**
    
    .. autosummary::
    
        Body.save
        Body.to_string
    
    **Basic plotting** (see plotting module for more options)
    
    .. autosummary::
        
       Body.plot2D
       Body.plot3D
        
       
       
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
    
    def add_pbdeps(self,pbdep):
        """
        Add a list of dependable ParameterSets to the Body.
        """
        #-- add dependables to params
        parsed_refs = _parse_pbdeps(self,pbdep)
        #-- add columns to mesh
        if len(self.mesh):
            for ref in parsed_refs:
                dtypes = [('ld_{0}'.format(ref),'f8',(5,))]
                dtypes.append(('proj_{0}'.format(ref),'f8'))
                dtypes.append(('velo_{0}_'.format(ref),'f8',(3,)))
                dtypes.append(('_o_velo_{0}_'.format(ref),'f8',(3,)))
                dtypes = np.dtype(dtypes)
                new_cols = np.zeros(len(self.mesh),dtype=dtypes)
                for i,field in enumerate(new_cols.dtype.names):
                    self.mesh = pl.mlab.rec_append_fields(self.mesh,field,new_cols[field],dtypes=dtypes[i])
        logger.info('added pbdeps {0}'.format(parsed_refs))
        return parsed_refs
    
    def remove_pbeps(self,refs):
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
                #-- drop fields
                fields = 'ld_{0}'.format(ref),\
                         'lproj_{0}'.format(ref),\
                         'velo_{0}'.format(ref),\
                         '_o_velo_{0}'.format(ref)
                self.mesh = pl.mlab.rec_drop_fields(self.mesh,fields)
                logger.info('removed pbdeps {0}'.format(fields))
    
    def add_obs(self,obs):
        """
        Add a list of DataSets to the Body.
        
        @param obs: list of DataSets
        @type obs: list
        """
        #-- add data to params
        parsed_refs = _parse_obs(self,obs)
        logger.info('added obs {0}'.format(parsed_refs))
        return parsed_refs
    
    def remove_obs(self,refs):
        """
        Remove observation (and synthetic) ParameterSets from the Body.
        """
        refs = set(refs)
        for dep in self.params['obs']:
            syn = dep[:-3]+'syn'
            keys = set(self.params['obs'][dep].keys())
            intersect = list(keys & refs)
            while intersect:
                ref = intersect.pop()
                self.params['obs'][dep].pop(ref)
                if syn in self.params['syn'] and ref in self.params['syn'][syn]:
                    self.params['syn'][syn].pop(ref)
 
                #-- drop fields
                fields = 'ld_{0}'.format(ref),\
                         'lproj_{0}'.format(ref),\
                         'velo_{0}'.format(ref),\
                         '_o_velo_{0}'.format(ref)
                self.mesh = pl.mlab.rec_drop_fields(self.mesh,fields)
                logger.info('removed obs {0}'.format(ref))
        
    
    def remove_mesh(self):
        self.mesh = np.zeros(0,dtype=self.mesh.dtype)
    
    @decorators.parse_ref
    def prepare_reflection(self,ref=None):
        """
        Prepare the mesh to handle reflection from an other body.
        
        We only need one extra column with the incoming flux divided by pi
        to account for isotropic scattering. Doppler beaming and such should
        be taken into account in the reflection algorithm. In the isotropic case,
        reflections is aspect indepedent.
        
        If you want to do something non-isotropic, you'd better do some kind of
        raytracing I guess. This requires a different approach than the one
        implemented here.
        """
        for iref in ref:
            field = 'refl_{}'.format(iref)
            if field in self.mesh.dtype.names: continue
            dtypes = np.dtype([(field,'f8')])
            new_cols = np.zeros(len(self.mesh),dtype=np.dtype(dtypes))
            self.mesh = pl.mlab.rec_append_fields(self.mesh,field,new_cols[field])
            logger.info('added reflection column for pbdep {}'.format(iref))
    
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
                
    
    #def get_parset(self,ref=None,context=None,type='pbdep',category=None):
        #"""
        #Return the parameter set with the given ref from the C{params}
        #dictionary attached to the Body.
        
        #If ref is None, return the parameterSet of the body itself.
        #If ref is an integer, return the "n-th" parameterSet.
        #If ref is a string, return the parameterSet with the ref matching
        #the string.
        
        #returns parset and its ref (in reverse order)
        #"""
        #if ref is None or ref=='__bol':
            #logger.info("Requested bolometric parameterSet")
            #return list(self.params.values())[0],'__bol'
        #else:
            #counter = 0
            #subtypes = subtype and [subtype] or self.params[type].keys()
            #for itype in subtypes:
                ##itype += type[-3:] # we want here subtype=='lc'-->'lcdep' or 'lcsyn'
                #for ips in self.params[type][itype]:
                    #ps = self.params[type][itype][ips]
                    #is_ref = ('ref' in ps) and (ps['ref']==ref)
                    #is_number = counter==ref
                    #if is_ref or is_number:
                        #logger.info("Requested parset ref={}, type={}, subtype={} and found ref={}, context={}".format(ref,type,subtype,ps['ref'],ps.get_context()))
                        #return ps,ps['ref']
                    #counter += 1
            #logger.info("Requested parset ref={}, type={}, subtype={} but nothing was found".format(ref,type,subtype))
            #return None,None
    
    
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
        try:
            deltaz = self.get_distance('Rsol')
        except AttributeError:
            if len(self.params.values()) and 'distance' in self.params.values()[0]:
                deltaz = self.params.values()[0].request_value('distance','Rsol')
                #ps['angular_coordinates'] =
                #d = conversions.convert(distance[1],'m',distance[0]) 
                #x = 2*np.arctan(x/(2*d))/pi*180*3600 
                #y = 2*np.arctan(y/(2*d))/pi*180*3600 
                #z = 2*np.arctan(z/(2*d))/pi*180*3600 
            else:
                deltaz = 0.
        origin = np.array([0,0,deltaz])
        
        #-- on-sky-coordinates
        if deltaz>0:
            try:
                #coords = self.get_sky_coordinates()
                coords = 0.,0.
            except AttributeError:
                pass
        
        
        #-- 1-3.  Geometric barycentre and photocentric barycentre
        ps['coordinates'] = np.average(self.mesh['center'],weights=wsize,axis=0)+origin,'Rsol'
        ps['photocenter'] = np.average(self.mesh['center'],weights=wflux*wsize,axis=0)+origin,'Rsol'
        ps['velocity']    = np.average(self.mesh['velo___bol_'],axis=0),'Rsol/d'
        ps['distance']    = sqrt(ps['coordinates'][0]**2+ps['coordinates'][1]**2+ps['coordinates'][2]**2),'Rsol'
        #-- 4.  mass: try to call the function specifically designed to compute
        #       the mass, if the class implements it.
        try:
            ps['mass'] = self.get_mass('Msol'),'Msol'
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
    
    def reset_mesh(self):
        """
        Reset the mesh to its original position
        """
        columns = self.mesh.dtype.names
        for column in columns:
            if column[:3]=='_o_' and column[3:] in columns:
                self.mesh[column[3:]] = self.mesh[column]
        self.mesh['partial'] = False
        self.mesh['visible'] = False
        self.mesh['hidden'] = True
        logger.info('reset mesh to original position')
    
    
    def update_mesh(self,subset=None):
        """
        Update the mesh for a subset of triangles or the whole mesh
        """
        if subset is None:
            subset = np.ones(len(self.mesh),bool)
        #-- cut out the part that needs to be updated
        old_mesh = self.mesh[subset].copy()
        #-- remember which arguments were used to create the original mesh
        mesh_args = self.subdivision['mesh_args']
        mesh_args,scale = mesh_args[:-1],mesh_args[-1]
        #logger.info('updating %d/%d triangles in mesh with args %s (scale=%s)'%(sum(subset),len(self.mesh),str(mesh_args),str(scale)))
        logger.info('updating some triangles in mesh with args %s (scale=%s)'%(str(mesh_args),str(scale)))
        #-- then reproject the old coordinates. We assume they are fairly
        #   close to the real values.
        
        
    
        #-- C or Python implementation:
        if True:
            select = ['_o_center','size','_o_triangle','_o_normal_']
            old_mesh_table = np.column_stack([old_mesh[x] for x in select])/scale
            old_mesh_table = marching.creproject(old_mesh_table,*mesh_args)*scale
            
            # Check direction of normal
            cosangle = coordinates.cos_angle(old_mesh['_o_center'],
                                             old_mesh_table[:,13:16],axis=1)
            sign = np.where(cosangle<0,-1,1).reshape((-1,1))
            
            
            for prefix in ['_o_','']:
                old_mesh[prefix+'center'] = old_mesh_table[:,0:3]
                old_mesh[prefix+'triangle'] = old_mesh_table[:,4:13]
                old_mesh[prefix+'normal_'] = sign*old_mesh_table[:,13:16]
            
            
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
                
        #-- normals are updated, but sizes are not.
        self.compute_sizes(prefix='_o_')
        #-- insert the updated values in the original mesh
        #mlab.figure()
        #self.plot3D(normals=True)    
        
        self.mesh[subset] = old_mesh
        
        #mlab.figure()
        #self.plot3D(normals=True)    
        #mlab.show()
        
    
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
        logger.info("subdividing: type {:d} via {:s}".format(subtype,algorithm))
        #-- subidivde the partially visible triangles
        partial = self.mesh['partial']
        subdivided = subdivision.subdivide(self.mesh[partial],prefix=prefix,
              threshold=threshold,algorithm=algorithm)
        #-- orientate the new triangles in the universe (rotation + translation)
        if len(subdivided):
            #-- replace old triangles with newly subdivided ones, but remember the
            #   old ones if this is the first time we subdivide the mesh
            if self.subdivision['orig'] is None:
                self.subdivision['orig'] = self.mesh.copy()
            self.mesh = np.hstack([self.mesh[-partial],subdivided])
            if subtype==1:
                #self.update_mesh(self.mesh['partial'])
                self.rotate_and_translate(subset=self.mesh['partial'])
                logger.info('rotated subdivided mesh')
        return len(subdivided)
    
    
    def unsubdivide(self):
        """
        Revert to the original, unsubdivided mesh.
        """
        if self.subdivision['orig'] is None:
            logger.info('nothing to unsubdivide')
        else:
            self.mesh = self.subdivision['orig']
            logger.info("restored mesh from subdivision")
        self.subdivision['orig'] = None
    
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
    def lc(self,correct_oversampling=1,ref='alllcdep',time=None):
        """
        Compute projected intensity and add results to the pbdep ParameterSet.
        
        """
        #-- don't bother if we cannot do anything...
        if hasattr(self,'params') and 'pbdep' in self.params:
            if not ('lcdep' in self.params['pbdep']): return None
            #-- compute the projected intensities for all light curves.
            for lbl in ref:
                base,lbl = self.get_parset(ref=lbl,type='syn')
                proj_intens = self.projected_intensity(ref=lbl)
                base['time'].append(self.time)
                base['flux'].append(proj_intens)
                #-- correct for oversampling
                if correct_oversampling>1:
                    #-- the timestamps (should just be the middle value but anyway)
                    mytime = np.mean(base['time'][-correct_oversampling:])
                    base['time'] = base['obs'][:-correct_oversampling:]
                    base['time'].append(mytime)
                    #-- the fluxes should be averaged
                    myintens = np.mean(base['flux'][-correct_oversampling:])
                    base['flux'] = base['flux'][:-correct_oversampling]
                    base['flux'].append(myintens)
                
    @decorators.parse_ref
    def rv(self,correct_oversampling=1,ref='allrvdep',time=None):
        """
        Compute integrated radial velocity and add results to the pbdep ParameterSet.
        """
        #-- don't bother if we cannot do anything...
        if hasattr(self,'params') and 'pbdep' in self.params:
            if not ('rvdep' in self.params['pbdep']): return None
            #-- compute the projected intensities for all light curves.
            for lbl in ref:
                base,lbl = self.get_parset(ref=lbl,type='syn')
                proj_velo = self.projected_velocity(ref=lbl)
                base['time'].append(self.time)
                base['rv'].append(proj_velo)
    
    @decorators.parse_ref
    def ps(self,ref='alllcdep',time=None):
        """
        Compute point-source representation of Body.
        """
        if hasattr(self,'params') and 'pbdep' in self.params:
            #-- compute the projected intensities for all light curves.
            for lbl in ref:
                myps = self.as_point_source(ref=lbl)
                #base,lbl = self.get_parset(ref=lbl,type='syn')
                for qualifier in myps:
                    self.params['pbdep']['psdep'][lbl]['syn']['time'].append(self.time)
                    
    #@decorators.parse_ref
    #def sp(self,wavelengths=None,ref='allspdep',sigma=5.,depth=0.4,time=None):
        #"""
        #Compute spectrum and add results to the pbdep ParameterSet.
        #"""
        ##-- don't bother if we cannot do anything...
        #if hasattr(self,'params') and 'pbdep' in self.params:
            #if not ('spdep' in self.params['pbdep']): return None
            ##-- compute the spectrum for all references
            #for lbl in ref:
                #base,lbl = self.get_parset(ref=lbl,type='syn')
                #wavelengths_,specflux,cont = observatory.make_spectrum(self,ref=lbl,wavelengths=wavelengths,sigma=sigma,depth=depth)
                #base['time'].append(self.time)
                #base['wavelength'].append(wavelengths_)
                #base['flux'].append(specflux)
                #base['continuum'].append(cont)
        
    #@decorators.parse_ref
    #def pl(self, wavelengths=None, ref='allpldep', sigma=5.,depth=0.4, time=None):
        #"""
        #Compute Stokes profiles and add results to the pbdep ParameterSet.
        #"""
        ## Don't bother if we cannot do anything...
        #if hasattr(self,'params') and 'pbdep' in self.params:
            #if not ('pldep' in self.params['pbdep']):
                #return None
            
            ## Compute the Stokes profiles for all references
            #for lbl in ref:
                
                ## Compute the Stokes profiles for this reference
                #base, lbl = self.get_parset(ref=lbl, type='syn')
                #output = observatory.stokes(self,ref=lbl,
                         #wavelengths=wavelengths,sigma=sigma,depth=depth)
                
                ## If nothing was computed, continue on to the next
                #if output is None:
                    #continue
                
                ## Expand output and save it to the synthetic thing
                #wavelengths_, I, V, Q , U, cont = output
                
                #base['time'].append(self.time)
                #base['wavelength'].append(wavelengths_/10.)
                #base['flux'].append(I)
                #base['V'].append(V)
                #base['Q'].append(Q)
                #base['U'].append(U)
                #base['continuum'].append(cont)
    
        
    
    
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
       Body.compute_pblum_or_l3
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
    >>> for star in bb2:
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
    def __init__(self,list_of_bodies, obs=None, report_problems=False, 
                 solve_problems=False, **kwargs):
        """
        Initialise a BodyBag.
        
        Extra keyword arguments can be Parameters that go in the C{self.params}
        dictionary of the BodyBag.
        
        @param list_of_bodies: list of bodies
        @type list_of_bodies: list
        """
        # We definitely need signals and a label, even if it's empty
        self.signals = {}
        self.label = None
        
        # Make sure the list of bodies is a list
        if not isinstance(list_of_bodies, list):
            list_of_bodies = [list_of_bodies]
        self.bodies = list_of_bodies
        
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
        
        # Also the _plot is a leftover from earlier days, this is deprecated
        self._plot = self.bodies[0]._plot
        
        # Process any extra keyword arguments: they can contain a label for the
        # bodybag, or extra ParameterSets. We need to set a label for a BodyBag
        # if it's a binary, so that we can figure out which component it is
        for key in kwargs:
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
        #-- make sure to pass on calls to builtin functions immediately to the
        #   bodybag.
        if name.startswith('__') and name.endswith('__'):
            return super(BodyBag,self).__getattr__(name)        
        #-- all other functions needs to pass by CallInstruct to see if they
        #   can be called from BodyBag or from each object individually.
        else:
            return CallInstruct(name,self.bodies)
    
    def __iter__(self):
        """
        Makes the class iterable.
        """
        for param in list(self.params.values()):
            yield param
        for body in self.bodies:
            yield body
    
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
    
    def __getitem__(self,key):
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
        #-- via slicing
        if isinstance(key,slice):
            return BodyBag([self[ii] for ii in range(*key.indices(len(self)))])
        #-- via an integer
        elif isinstance(key,int):
            return self.bodies[key]
        else:
            #-- try to make the input an array
            try:
                key = np.array(key)
            except:
                raise IndexError("Cannot use instance of type {} for indexing".format(type(key)))
            #-- integer array slicing
            if key.dtype==np.dtype(int):
                return BodyBag([self[ii] for ii in key])
            #-- boolean array slicing
            elif key.dtype==np.dtype(bool):
                return BodyBag([self[ii] for ii in range(len(key)) if key[ii]])
            #-- that's all I can come up with
            else:
                raise IndexError("Cannot use arrays of type {} for indexing".format(key.dtype))
    
    def __str__(self):
        return self.to_string()
    
    def fix_mesh(self):
        """
        Make sure all bodies in a list have the same mesh columns.
        """
        #-- here, we check which columns are missing from each Body's
        #   mesh. If they are missing, we simply add them and copy
        #   the contents from the original mesh.
        logger.info("Preparing mesh")
        names = list(self.bodies[0].mesh.dtype.names)
        descrs = self.bodies[0].mesh.dtype.descr
        for b in self.bodies[1:]:
            descrs_ = b.mesh.dtype.descr
            for descr in descrs_:
                if descr[0] in names: continue
                descrs.append(descr)                    
                names.append(descr[0])
        dtypes = np.dtype(descrs)
        for b in self.bodies:
            N = len(b.mesh)
            new_mesh = np.zeros(N,dtype=dtypes)
            if N:
                cols_to_copy = list(b.mesh.dtype.names)
                for col in cols_to_copy:
                    new_mesh[col] = b.mesh[col]
                #new_mesh[cols_to_copy] = b.mesh[cols_to_copy]
            b.mesh = new_mesh
    
    def remove_mesh(self):
        for body in self.bodies:
            body.remove_mesh()
    
    def add_obs(self,obs):
        _parse_obs(self,obs)
    
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
    
    def set_time(self, *args, **kwargs):
        """
        Set the time of all the Bodies in the BodyBag.
        """
        for body in self.bodies:
            body.set_time(*args, **kwargs)
        if 'orbit' in self.params:
            #-- once we have the mesh, we need to place it into orbit
            keplerorbit.place_in_binary_orbit(self, args[0])
    
    def set_label(self,label):
        try:
            comp = self.get_component()
            if comp==0:
                self.params['orbit']['c1label'] = label
            elif comp==1:
                self.params['orbit']['c2label'] = label
        except Exception as msg:
            logger.error(str(msg))
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
        # being built from the ones in the bodies list. E.g. interferometry can
        # only be computed of the whole system, since the total Fourier
        # transform is not the sum of the component Fourier transforms.
        #   .... euhh... it kind of is! Anyway...
        if kwargs.get('category', 'lc') == 'if':
            total_results = super(BodyBag, self).get_synthetic(*args, **kwargs)
            if total_results:
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
    
    def get_logp(self):
        logp,chi2,N = super(BodyBag,self).get_logp()
        for body in self.bodies:
            this_logp,this_chi2,this_N = body.get_logp()
            logp = logp + this_logp
            chi2 = chi2 + this_chi2
            N = N + this_N
        return logp, chi2, N
    
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
        if 'orbit' in self.params:
            distance = self.params['orbit'].request_value('distance','Rsol')
        elif 'orbit' in self.bodies[0].params:
            distance = self.bodies[0].params['orbit'].request_value('distance','Rsol')
        else:
            distance = 0
            logger.warning("Don't know distance")
        coords[2] += distance
        if only_coords:
            return coords
        else:
            return dict(coordinates=coords)
    
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
    def __new__(self,objs,orbit,solve_problems=True,**kwargs):
        """
        Parameter objs needs to a list, perhaps [None, object]  or [object, None]
        if you only want to create a BinaryBag of one object.
        
        To do: if one of the components is a star, optionally morph it to
        BinaryRocheStar
        """
        if len(objs)>2:
            raise ValueError("Binaries consist of a maximum of two objects ({} given)".format(len(objs)))
        
        system = []
        
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
                    iobject.set_label(ilabel)
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
            return BodyBag(system,solve_problems=solve_problems,**kwargs)
        else:
            return system[0]
        

class AccretionDisk(PhysicalBody):
    """
    Flaring Accretion Disk.
    
    The implementation of the L{AccretionDisk} follows closely the descriptions
    presented in the papers of U{Copperwheat et al. (2010) <http://adsabs.harvard.edu/abs/2010MNRAS.402.1824C>} and U{Wood et al. (1992) <http://adsabs.harvard.edu/abs/1992ApJ...393..729W>}.
    
    There is no limb-darkening included yet.
    
    """
    def __init__(self,accretion_disk,pbdep=None,reddening=None,**kwargs):
        """
        Initialize a flaring accretion disk.
        
        The only parameters need for the moment are the disk parameters.
        In the future, also the details on the mesh creation should be given.
        """
        super(AccretionDisk,self).__init__(dim=3,**kwargs)
        self.params['disk'] = accretion_disk
        self.params['pbdep'] = OrderedDict()
        #-- add interstellar reddening (if none is given, set to the default,
        #   this means no reddening
        if reddening is None:
            reddening = parameters.ParameterSet(context='reddening:interstellar')
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

    def compute_mesh(self,radial=20,angular=50):
        Rin = self.params['disk'].get_value('rin','Rsol')
        Rout = self.params['disk'].get_value('rout','Rsol')
        height = self.params['disk'].get_value('height','Rsol')
        ld_law = 5
        ldbol_law = 5
        if not 'logg' in self.mesh.dtype.names:
            lds = [('ld___bol','f8',(5,)),('proj___bol','f8')]
            for pbdeptype in self.params['pbdep']:
                for ipbdep in self.params['pbdep'][pbdeptype]:
                    ipbdep = self.params['pbdep'][pbdeptype][ipbdep]
                    lds.append(('ld_{0}'.format(ipbdep['ref']),'f8',(5,)))
                    lds.append(('proj_{0}'.format(ipbdep['ref']),'f8'))
                    lds.append(('velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
                    lds.append(('_o_velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
            dtypes = np.dtype(self.mesh.dtype.descr + \
                     [('logg','f8'),('teff','f8')] + lds)
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
        Temperature is calculated according to [Wood1992]_
        
        .. math::
        
            T_\mathrm{eff}(r)^4 = (G M_\mathrm{wd} \dot{M}) / (8\pi r^3) (1-b\sqrt{R_\mathrm{in}/r}) 
        
        An alternative formula might be from Frank et al. 1992 (p. 78):
        
        .. math::
            T_\mathrm{eff}r)^4 = \frac{W^{0.25} 3G M_\mathrm{wd} \dot{M}}{8 \pi r^3 \sigma}  (1-\sqrt{\frac{R_*}{r}}) \left(\frac{r}{R_*}\right)^{0.25\beta}
        
        with :math:`\beta=-0.75` and :math:`W=1.0` for the standard model.
        """
        r = coordinates.norm(self.mesh['_o_center'],axis=1)*constants.Rsol
        Mdot = self.params['disk'].get_value('dmdt','kg/s')
        M_wd = self.params['disk'].get_value('mass','kg')
        Rin = self.params['disk'].get_value('rin','m')
        b = self.params['disk']['b']
        sigTeff4 = constants.GG*M_wd*Mdot/(8*pi*r**3)*(1-b*sqrt(Rin/r))
        sigTeff4[sigTeff4<0] = 1e-1 #-- numerical rounding (?) can do weird things
        self.mesh['teff'] = (sigTeff4/constants.sigma)**0.25
        
    @decorators.parse_ref
    def intensity(self,*args,**kwargs):
        """
        Calculate local intensity and limb darkening coefficients.
        """
        ref = kwargs.pop('ref',['all'])
        parset_isr = self.params['reddening']
        #-- now run over all labels and compute the intensities
        for iref in ref:
            parset_pbdep,ref = self.get_parset(ref=iref,type='pbdep')
            limbdark.local_intensity(self,parset_pbdep,parset_isr)
            
    def projected_intensity(self,los=[0.,0.,+1],ref=0,method='numerical',with_partial_as_half=True):
        """
        Calculate local intensity.
        """
        if method!='numerical':
            raise ValueError("Only numerical computation of projected intensity of AccretionDisk available")
        idep,ref = self.get_parset(ref=ref,type='pbdep')
        ld_func = idep['ld_func']
        proj_int = limbdark.projected_intensity(self,method=method,
                ld_func=ld_func,ref=ref,with_partial_as_half=with_partial_as_half)
        return proj_int
    
    def set_time(self,time, ref='all'):
        self.reset_mesh()
        if self.time is None:
            self.compute_mesh()
            self.surface_gravity()
            self.temperature()
            self.intensity()
        self.time = time
        
        
        
class Star(PhysicalBody):
    """
    Body representing a Star.
    
    Construct a simple star with the default parameters representing the Sun
    (sort of, don't start nitpicking):
    
    >>> star_pars = parameters.ParameterSet(frame='phoebe',context='star',add_constraints=True,label='mystar')
    >>> lcdep1 = parameters.ParameterSet(frame='phoebe',context='lcdep',ref='mylc')
    >>> mesh = parameters.ParameterSet(frame='phoebe',context='mesh')
    >>> star = Star(star_pars,mesh,pbdep=[lcdep1])
    
    We initialized the star with these parameters::
    
        # print star_pars
              teff 5777.0           K phoebe Effective temperature
            radius 1.0           Rsol phoebe Radius
              mass 1.0           Msol phoebe Stellar mass
               atm blackbody       -- phoebe Bolometric Atmosphere model
         rotperiod 22.0             d phoebe Equatorial rotation period
             gravb 1.0             -- phoebe Bolometric gravity brightening
              incl 90.0           deg phoebe Inclination angle
          distance 10.0            pc phoebe Distance to the star
           surface roche           -- phoebe Type of surface
        irradiator False           -- phoebe Treat body as irradiator of other objects
             label mystar          -- phoebe Name of the body
          ld_func uniform         -- phoebe Bolometric limb darkening model
                cl [1.0]           -- phoebe Bolometric limb darkening coefficients
          surfgrav 274.351532944  n/a constr constants.GG*{mass}/{radius}**2    
    
    
    Upon initialisation, only the basic parameters are set, nothing is computed.
    We need to compute the mesh. We can do this manually through the function
    L{Star.compute_mesh}, but it is easiest just to set the time, and let the
    Body take care of itself:
    
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
    
    Again with the plots:
    
    >>> p = mlab.figure(bgcolor=(0.5,0.5,0.5))
    >>> p = star.plot3D(select='teff',colormap='spectral')
    >>> p = mlab.colorbar()
    
    ]include figure]]images/universe_star_0003.png]
    
    >>> out = observatory.image(star,savefig=False)
    
    ]include figure]]images/universe_star_0004.png]
    
    """
    def __init__(self, star, mesh, reddening=None, circ_spot=None, puls=None,
                 magnetic_field=None, pbdep=None, obs=None, **kwargs):
        """
        Initialize a star.
        
        What needs to be done? We'll, sit down and let me explain. Are you
        sitting down? Yes, then let's begin our journey through the birth
        stage of a Star. In many ways, the birth of a Star can be regarded as
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
        
        # Prepare basic parameterSets and Ordered dictionaries
        self.params['star'] = star
        self.params['mesh'] = mesh
        self.params['pbdep'] = OrderedDict()
        self.params['obs'] = OrderedDict()
        self.params['syn'] = OrderedDict()
        
        # Shortcut to make a binaryStar
        if 'orbit' in kwargs:
            self.params['orbit'] = kwargs.pop('orbit')
            
        # Add interstellar reddening (if none is given, set to the default, this
        # means no reddening
        if reddening is None:
            reddening = parameters.ParameterSet(context='reddening:interstellar')
        self.params['reddening'] = reddening
        
        # Add spot parameters when applicable
        if circ_spot is not None:
            if not isinstance(circ_spot, list):
                to_add = [circ_spot]
            else:
                to_add = circ_spot
            self.params['circ_spot'] = to_add
            
        # Add pulsation parameters when applicable
        if puls is not None:
            if not isinstance(puls, list):
                to_add = [puls]
            else:
                to_add = puls
            self.params['puls'] = to_add
            
        # Add magnetic field parameters when applicable
        if magnetic_field is not None:
            self.params['magnetic_field'] = magnetic_field
        
        # Add the parameters to compute dependables
        if pbdep is not None:
            _parse_pbdeps(self, pbdep)
        
        # Add the parameters from the observations
        if obs is not None:
            _parse_obs(self, obs)
        
        # Check for leftover kwargs and report to the user
        if kwargs:
            logger.warning("Unused keyword arguments {} upon initialization of Star".format(kwargs.keys()))
        
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
        Get the label of the Body.
        
        @return: label of the Star
        @rtype: str
        """
        return self.params['star']['label']
    
    
    def surface_gravity(self):
        """
        Calculate local surface gravity
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
        Calculate local temperature.
        """
        # If the gravity brightening law is not specified, use 'Zeipel's
        if 'gravblaw' in self.params['star']:
            gravblaw = self.params['star']['gravblaw']
        else:
            gravblaw = 'zeipel'
        
        # Compute temperature
        getattr(roche,'temperature_{}'.format(gravblaw))(self)
        
        # Perhaps we want to add spots.
        self.add_spots(time)
    
    
    def abundance(self, time):
        """
        Set the abundance.
        """
        self.mesh['abun'] = list(self.params.values())[0]['abun']
    
    
    def magnetic_field(self):
        """
        Calculate the magnetic field.
        
        Problem: when the surface is deformed, I need to know the value of
        the radius at the magnetic pole! Or we could just interpret the
        polar magnetic field as the magnetic field strength in the direction
        of the magnetic axes but at a distance of 1 polar radius....
        """
        # Dipolar field:
        parset = self.params['magnetic_field']
        beta = parset.get_value('beta', 'rad')
        phi0 = parset.get_value('phi0', 'rad')
        Bpolar = parset.get_value('Bpolar')
        R = self.params.values()[0].get_value('radius')
        r_ = self.mesh['_o_center'] / R
        
        m_ = np.array([np.sin(beta) * np.cos(phi0) - 0.0*np.sin(phi0),
                       np.sin(beta) * np.sin(phi0) + 0.0*np.cos(phi0),
                       np.cos(beta)])
        dotprod = np.dot(m_, r_.T).reshape(-1, 1)
        B =     (3*dotprod    *r_ - m_)
        B = B / 2.0 * Bpolar
        self.mesh['_o_B_'] = B
        self.mesh['B_'] = self.mesh['_o_B_']
        logger.info("Added magnetic field with Bpolar={}G, beta={} deg".format(Bpolar, beta/pi*180))
        logger.info("Maximum B-field on surface = {}G".format(coordinates.norm(B, axis=1).max()))
    
    
    @decorators.parse_ref
    def intensity(self, ref='all'):
        """
        Calculate local intensity and limb darkening coefficients.
        """
        #-- now run over all labels and compute the intensities
        parset_isr = self.params['reddening']
        for iref in ref:
            parset_pbdep, ref = self.get_parset(ref=iref, type='pbdep')
            limbdark.local_intensity(self, parset_pbdep, parset_isr)
        
    
    @decorators.parse_ref
    def velocity(self, time=None, ref=None):
        """
        Calculate the velocity of each surface via the rotational velocity
        """
        if time is None:
            time = self.time
        #-- rotational velocity: first collect some information
        omega_rot = 1./self.params['star'].request_value('rotperiod','d')
        omega_rot = np.array([0.,0.,-omega_rot])
        logger.info('Calculating rotation velocity (Omega={:.3f} rad/d)'.format(omega_rot[-1]*2*pi))
        if 'diffrot' in self.params['star'] and self.params['star']['diffrot']!=0:
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
        inclin = self.params['star'].request_value('incl','rad')
        #-- the velocity is the cross product of the centers with
        #   the rotation vector pointed in the Z direction.
        velo_rot = 2*pi*np.cross(self.mesh['_o_center'],omega_rot) #NX3 array
        #-- for logging purposes, we compute the magnitude of the velocity
        #velo_mag = sqrt((velo_rot**2).sum(axis=1))
        #print "... velocity between %.3g and %.3g Rsol/d"%(velo_mag.min(),velo_mag.max())
        #-- total
        #-- We need to rotate the velocities so that they are in line with the
        #   current configuration
        #velo_rot_ = fgeometry.rotate3d_orbit_conv(velo_rot,(0,inclin,0),[0,0,0],'YXZ')
        for iref in ['__bol']:#ref:
            ps,iref = self.get_parset(iref)
            self.mesh['_o_velo_'+iref+'_'] = velo_rot
            self.mesh['velo_'+iref+'_'] = velo_rot#_
        #-- and we need the systemic velocity too...
        self.mesh['velo___bol_'][:,2] = self.mesh['velo___bol_'][:,2] #+ self.params['star'].request_value('vgamma','Rsol/d')
        v = sqrt(velo_rot[:,0]**2+velo_rot[:,1]**2+velo_rot[:,2]**2)
        
    
    def projected_intensity(self,los=[0.,0.,+1],ref=0,method=None,with_partial_as_half=True):
        """
        Calculate local intensity.
        
        We can speed this up if we compute the local intensity first, keep track of the limb darkening
        coefficients and evaluate for different angles. Then we only have to do a table lookup once.
        """
        idep,ref = self.get_parset(ref=ref,type='pbdep')
        if method is None:
            method = 'method' in idep and idep['method'] or 'numerical'
        ld_func = idep['ld_func']
        l3 = idep.get('l3',0.)
        pblum = idep.get('pblum',1.0)
        proj_int = limbdark.projected_intensity(self,method=method,
                ld_func=ld_func,ref=ref,with_partial_as_half=with_partial_as_half)
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
        if time is None:
            time = self.time
        
        #-- relevant stellar parameters
        rotfreq = 1./self.params['star'].request_value('rotperiod','d')
        R = self.params['star'].request_value('radius','m')
        M = self.params['star'].request_value('mass','kg')    
        
        #-- prepare extraction of pulsation parameters
        freqs = []
        freqs_Hz = []
        phases = []
        ampls = []
        ls = []
        ms = []
        deltaTs = []
        deltags = []
        ks = []
        spinpars = []
        
        #-- extract pulsation parameters, depending on their scheme
        for i,pls in enumerate(self.params['puls']):
            #-- extract information on the mode
            scheme = pls.get_value('scheme')
            l = pls.get_value('l')
            m = pls.get_value('m')
            k_ = pls.get_value('k')
            freq = pls.get_value('freq','cy/d')
            freq_Hz = freq / (24.*3600.)
            ampl = pls.get_value('ampl')
            deltaT = pls.get_value('amplteff')*np.exp(1j*2*pi*pls.get_value('phaseteff'))
            deltag = pls.get_value('amplgrav')*np.exp(1j*2*pi*pls.get_value('phasegrav'))
            phase = pls.get_value('phase')
            omega = 2*pi*freq_Hz
            k0 = constants.GG*M/omega**2/R**3    
            #-- if the pulsations are defined in the scheme of the traditional
            #   approximation, we need to expand the single frequencies into many.
            #   indeed, the traditional approximation approximates a mode as a
            #   linear combination of modes with different degrees.        
            if scheme=='traditional approximation':
                #-- extract some info on the B-vector
                bvector = pls.get_value('trad_coeffs')
                N = len(bvector)
                ljs = np.arange(N)
                for lj,Bjk in zip(ljs,bvector):
                    if Bjk==0: continue
                    #if lj>50: continue
                    freqs.append(freq)
                    freqs_Hz.append(freq_Hz)
                    ampls.append(Bjk*ampl)
                    phases.append(phase)
                    ls.append(lj)
                    ms.append(m)
                    deltaTs.append(Bjk*deltaT)
                    deltags.append(Bjk*deltag)
                    ks.append(k0)
                    spinpars.append(0.) # not applicable
            elif scheme=='nonrotating' or scheme=='coriolis':
                if scheme=='coriolis' and l>0:
                    spinpar = rotfreq/freq
                    Cnl = pls.get_value('ledoux_coeff')
                    k = k0 + 2*m*spinpar*((1.+k0)/(l**2+l)-Cnl)
                    logger.info('puls: adding Coriolis (rot=%.3f cy/d) effects for freq %.3f cy/d (l,m=%d,%d): ah/ar=%.3f, spin=%.3f'%(rotfreq,freq,l,m,k,spinpar))
                else:
                    spinpar = 0.
                    k = k_#k0
                    logger.info('puls: no Coriolis (rot=%.3f cy/d) effects for freq %.3f cy/d (l,m=%d,%d): ah/ar=%.3f, spin=0'%(rotfreq,freq,l,m,k))
                freqs.append(freq)
                freqs_Hz.append(freq_Hz)
                ampls.append(ampl)
                phases.append(phase)
                ls.append(l)
                ms.append(m)
                deltaTs.append(deltaT)
                deltags.append(deltag)
                ks.append(k)
                spinpars.append(spinpar)
            else:
                raise ValueError('Pulsation scheme {} not recognised'.format(scheme))
            
        #-- then add displacements due to pulsations. When computing the centers,
        #   we also add the information on teff and logg
        #index = np.array([2,0,1])
        #index_inv = np.array([1,2,0])
        index = np.array([1,0,2])
        index_inv = np.array([1,0,2])
        puls_incl = self.params['puls'][0].get_value('incl','rad')
        r1,phi1,theta1 = coordinates.cart2spher_coord(*self.mesh['_o_triangle'][:,0:3].T[index])
        r2,phi2,theta2 = coordinates.cart2spher_coord(*self.mesh['_o_triangle'][:,3:6].T[index])
        r3,phi3,theta3 = coordinates.cart2spher_coord(*self.mesh['_o_triangle'][:,6:9].T[index])
        r4,phi4,theta4 = coordinates.cart2spher_coord(*self.mesh['_o_center'].T[index])
        r1,theta1,phi1,vr1,vth1,vphi1 = pulsations.surface(r1,theta1,phi1,time,ls,ms,freqs,phases,spinpars,ks,ampls)        
        r2,theta2,phi2,vr2,vth2,vphi2 = pulsations.surface(r2,theta2,phi2,time,ls,ms,freqs,phases,spinpars,ks,ampls)
        r3,theta3,phi3,vr3,vth3,vphi3 = pulsations.surface(r3,theta3,phi3,time,ls,ms,freqs,phases,spinpars,ks,ampls)
        r4,theta4,phi4,vr4,vth4,vphi4,teff,logg = pulsations.observables(r4,theta4,phi4,
                     self.mesh['teff'],self.mesh['logg'],time,ls,ms,freqs,phases,
                     spinpars,ks,ampls,deltaTs,deltags)
        self.mesh['triangle'][:,0:3] = np.array(coordinates.spher2cart_coord(r1,phi1,theta1))[index_inv].T
        self.mesh['triangle'][:,3:6] = np.array(coordinates.spher2cart_coord(r2,phi2,theta2))[index_inv].T
        self.mesh['triangle'][:,6:9] = np.array(coordinates.spher2cart_coord(r3,phi3,theta3))[index_inv].T
        #for iref in ref:
        #    ps,iref = self.get_parset(iref)
        #    self.mesh['velo_%s_'%(iref)] += np.array(coordinates.spher2cart((r4,phi4,theta4),(vr4,vphi4,vth4)))[index_inv].T
        self.mesh['velo___bol_'] += np.array(coordinates.spher2cart((r4,phi4,theta4),(vr4,vphi4,vth4)))[index_inv].T
        self.mesh['center'] = np.array(coordinates.spher2cart_coord(r4,phi4,theta4))[index_inv].T
        self.mesh['teff'] = teff
        self.mesh['logg'] = logg
        logger.info("puls: computed pulsational displacement, velocity and teff/logg field")
    
    def compute_mesh(self,time=None):
        """
        Compute the mesh
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
        ld_law = 5
        ldbol_law = 5
        if not 'logg' in self.mesh.dtype.names:
            lds = [('ld___bol','f8',(5,)),('proj___bol','f8')]
            for pbdeptype in self.params['pbdep']:
                for ipbdep in self.params['pbdep'][pbdeptype]:
                    ipbdep = self.params['pbdep'][pbdeptype][ipbdep]
                    lds.append(('ld_{0}'.format(ipbdep['ref']),'f8',(5,)))
                    lds.append(('proj_{0}'.format(ipbdep['ref']),'f8'))
                    lds.append(('velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
                    lds.append(('_o_velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
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
        logger.info('updating %d/%d triangles in mesh'%(sum(subset),len(self.mesh)))
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

    def set_time(self,time,ref='all'):
        """
        Set the time of the Star object.
        
        @param time: time
        @type time: float
        @param label: select columns to fill (i.e. bolometric, lcs)
        @type label: str
        """
        logger.info('===== SET TIME TO %.3f ====='%(time))
        #-- first execute any external constraints:
        self.preprocess(time)
        #-- this mesh is mostly independent of time! We collect some values
        #   that could be handy later on: inclination and rotation frequency
        rotperiod = self.params['star'].request_value('rotperiod','d')
        Omega_rot = 2*pi*time/rotperiod
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
        
        self.mesh['velo___bol_'][:,2] = self.mesh['velo___bol_'][:,2] - self.params['star'].request_value('vgamma','Rsol/d')
        if self.time is None or has_freq or has_spot:
            self.intensity(ref=ref)
        #-- remember the time... 
        self.time = time
        self.postprocess(time)
    
    
class BinaryRocheStar(PhysicalBody):    
    """
    Body representing a binary Roche surface.
    """
    
    def __init__(self,component,orbit=None,mesh=None,reddening=None,pbdep=None,obs=None,**kwargs):
        """
        Component: 0 is primary 1 is secondary
        """
        super(BinaryRocheStar,self).__init__(dim=3)
        #-- remember the values given
        self.params['component'] = component
        self.params['orbit'] = orbit
        self.params['mesh'] = mesh
        self.params['pbdep'] = OrderedDict()
        self.params['obs'] = OrderedDict()
        self.params['syn'] = OrderedDict()
        self.time = None
        #-- label the body
        self.label = self.params['component']['label']
        #-- add interstellar reddening (if none is given, set to the default,
        #   this means no reddening
        if reddening is None:
            reddening = parameters.ParameterSet(context='reddening:interstellar')
        self.params['reddening'] = reddening
        if pbdep is not None:
            _parse_pbdeps(self,pbdep)
        if obs is not None:
            _parse_obs(self,obs)
                
        #-- add common constraints
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
        
        init_mesh(self)
    
    def set_label(self,label):
        self.params['component']['label'] = label
    
    def get_label(self):
        """
        Get the label of the Body.
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
        Phi = self.params['component'].get_value('pot')
        com = self.params['orbit'].get_constraint('com','au') / a
        pivot = np.array([com,0,0]) # center-of-mass (should be multiplied by a!)
        T0 = self.params['orbit'].get_value('t0')
        scale = self.params['orbit'].get_value('sma','Rsol')
        
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
        r_pole = marching.projectOntoPotential((0,0,1e-5),'BinaryRoche',d,q,F,Phi).r
        r_pole_= np.linalg.norm(r_pole)
        r_pole = r_pole_*a
        g_pole = roche.binary_surface_gravity(0,0,r_pole,d*a,omega_rot/F,M1,M2,normalize=True)
        self.params['component'].add_constraint('{{r_pole}} = {0:.16g}'.format(r_pole))
        self.params['component'].add_constraint('{{g_pole}} = {0:.16g}'.format(g_pole))
        self.params['orbit'].add_constraint('{{d}} = {0:.16g}'.format(d*a))
                
        gridstyle = self.params['mesh'].context
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
        ld_law = 5
        ldbol_law = 5
        new_dtypes = []
        old_dtypes = self.mesh.dtype.names
        #-- check if the following required labels are in the mesh, if they
        #   are not, we'll have to add them
        required = [('ld___bol','f8',(5,)),('proj___bol','f8'),
                    ('logg','f8'),('teff','f8'),('abun','f8')]
        for req in required:
            if not req[0] in old_dtypes:
                new_dtypes.append(req)
        if 'pbdep' in self.params:
            for pbdeptype in self.params['pbdep']:
                for ipbdep in self.params['pbdep'][pbdeptype]:
                    ipbdep = self.params['pbdep'][pbdeptype][ipbdep]
                    if not 'ld_{0}'.format(ipbdep['ref']) in old_dtypes:
                        new_dtypes.append(('ld_{0}'.format(ipbdep['ref']),'f8',(5,)))
                        new_dtypes.append(('proj_{0}'.format(ipbdep['ref']),'f8'))
                        new_dtypes.append(('velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
                        new_dtypes.append(('_o_velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
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
        
    def conserve_volume(self,time,max_iter=10,tol=1e-6):
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
        #-- necessary values
        R = self.params['component'].request_value('r_pole','Rsol')
        sma = self.params['orbit'].request_value('sma','Rsol')
        e = self.params['orbit'].get_value('ecc')
        P = self.params['orbit'].get_value('period','s')
        q = self.params['orbit'].request_value('q')
        F = self.params['component'].request_value('syncpar')
        oldpot = self.params['component'].request_value('pot')
        M1 = self.params['orbit'].get_constraint('mass1','kg') # primary mass in solar mass
        M2 = q*M1
        #M2 = self.params['orbit'].get_constraint('mass2','kg') # secondary mass in solar mass
        component = self.get_component()+1
        
        #-- possibly we need to conserve the volume of the secondary component
        if component==2:
            q,oldpot = roche.change_component(q,oldpot)  
            M1,M2 = M2,M1
        
        pos1,pos2,d = get_binary_orbit(self,time)
        d_ = d*sma
        omega_rot = F * 2*pi/P # rotation frequency
        
        #-- keep track of the potential vs volume function to compute
        #   derivatives numerically
        potentials = []
        volumes = []
        
        #-- critical potential cannot be subceeded
        if F==1.:
            critpot = roche.calculate_critical_potentials(q,F,d)[0]
        else:
            critpot = 0.
        
        if max_iter>1:
            V1 = self.params['component'].request_value('volume')
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
        
        #-- keep parameters up-to-date
        g_pole = roche.binary_surface_gravity(0,0,R*constants.Rsol,d_*constants.Rsol,omega_rot/F,M1,M2,normalize=True)
        self.params['component'].add_constraint('{{r_pole}} = {0:.16g}'.format(R*constants.Rsol),do_run_constraints=False)
        self.params['component'].add_constraint('{{g_pole}} = {0:.16g}'.format(g_pole),do_run_constraints=False)
            
        
        #-- perhaps this was the secondary
        if component==2:
            q,oldpot = roche.change_component(q,oldpot)
        if max_iter>1:
            logger.info("volume conservation (V=%.6f<-->Vref=%.6f): changed potential Pot_ref=%.6f-->Pot_new%.6f)"%(V2,V1,self.params['component']['pot'],oldpot))
            #-- remember the new potential value
            self.params['component']['pot'] = oldpot
        else:
            logger.info("no volume conservation, reprojected onto instantaneous potential")
        
    def volume(self):
        """
        Compute volume of a convex mesh.
        """
        norm = coordinates.norm(self.mesh['_o_center'],axis=1)
        return np.sum(self.mesh['_o_size']*norm/3.)
        
    
    @decorators.parse_ref
    def intensity(self,ref='all'):
        """
        Calculate local intensity and limb darkening coefficients.
        """
        parset_isr = self.params['reddening']
        #-- now run over all labels and compute the intensities
        for iref in ref:
            parset_pbdep,ref = self.get_parset(ref=iref,type='pbdep')
            limbdark.local_intensity(self,parset_pbdep,parset_isr)
            
        
        
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
        dOmega_ = roche.binary_potential_gradient(self.mesh['_o_center'][:,0]/asol,
                                                  self.mesh['_o_center'][:,1]/asol,
                                                  self.mesh['_o_center'][:,2]/asol,
                                                  q,d,F,normalize=False) # component is not necessary as q is already from component
        Gamma_pole = roche.binary_potential_gradient(0,0,rp,q,d,F,normalize=True)        
        zeta = gp / Gamma_pole
        grav_local_ = dOmega_*zeta
        grav_local = coordinates.norm(grav_local_)
        
        #self.mesh['logg'] = conversions.convert('m/s2','[cm/s2]',grav_local)
        self.mesh['logg'] = np.log10(grav_local)+2.0
        logger.info("derived surface gravity: %.3f <= log g<= %.3f (g_p=%s and Rp=%s Rsol)"%(self.mesh['logg'].min(),self.mesh['logg'].max(),gp,rp*asol))

    def temperature(self,time=None):
        """
        Calculate local temperature.
        """
        roche.temperature_zeipel(self)
    
    
    def projected_velocity(self,los=[0,0,+1],ref=0,method=None):
        rvdep,ref = self.get_parset(ref=ref,type='pbdep')
        ld_func = rvdep.request_value('ld_func')
        method = 'numerical'
        return limbdark.projected_velocity(self,method=method,ld_func=ld_func,ref=ref)
        
    def projected_intensity(self,los=[0.,0.,+1],ref=0,method=None,with_partial_as_half=True):
        """
        Calculate local intensity.
        
        We can speed this up if we compute the local intensity first, keep track of the limb darkening
        coefficients and evaluate for different angles. Then we only have to do a table lookup once.
        """
        lcdep,ref = self.get_parset(ref)
        
        #-- get limb angles
        mus = self.mesh['mu']
        #-- To calculate the total projected intensity, we keep track of the
        #   partially visible triangles, and the totally visible triangles:
        keep = (mus>0) & (self.mesh['partial'] | self.mesh['visible'])
        mus = mus[keep]
        #-- negating the next array gives the partially visible things, that is
        #   the only reason for defining it.
        visible = self.mesh['visible'][keep]
        #-- compute intensity using the already calculated limb darkening coefficents
        logger.info('using limbdarkening law %s (%d vis)'%(lcdep['ld_func'],np.sum(keep)))
        Imu = getattr(limbdark,'ld_%s'%(lcdep['ld_func']))(mus,self.mesh['ld_'+ref][keep].T)*self.mesh['ld_'+ref][keep,-1]#*size   
        proj_Imu = mus*Imu
        if with_partial_as_half:
            proj_Imu[-visible] /= 2.0
        self.mesh['proj_'+ref] = 0.
        self.mesh['proj_'+ref][keep] = proj_Imu
        #-- take care of reflected light
        if 'refl_'+ref in self.mesh.dtype.names:
            proj_Imu += self.mesh['refl_'+ref][keep]
            logger.info("Projected intensity contains reflected light")
        proj_intens = self.mesh['size'][keep]*proj_Imu
        distance = self.params['orbit'].request_value('distance','Rsol')
        proj_intens = proj_intens.sum()/distance**2
        l3 = lcdep.get('l3',0.)
        pblum = lcdep.get('pblum',-1.0)
        
        if pblum >= 0:
            # This definition of passband luminosity should mimic the definition
            # of WD
            if not 'pblum' in self._clear_when_reset:
                passband_lum = luminosity(self,ref=ref)/ (100*constants.Rsol)**2
                passband_lum = passband_lum / distance**2
                self._clear_when_reset['pblum'] = passband_lum
            else:
                passband_lum = self._clear_when_reset['pblum']
            proj_intens = proj_intens * pblum / passband_lum + l3
        
        return proj_intens + l3
    
    
    
    def set_time(self,time,ref='all'):
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
        #-- rotate in 3D in orbital plane
        #   for zero-eccentricity, we don't have to recompute local quantities, and not
        #   even projected quantities (this should be taken care of on a higher level
        #   because other meshes can be in front of it
        #   For non-zero eccentricity, we need to recalculate the grid and recalculate
        #   local quantities
        e = self.params['orbit'].get_value('ecc')
        sma = self.params['orbit'].get_value('sma','Rsol')
        #-- there is a possibility to set to conserve volume or equipot
        #   IF eccentricity is zero, we never need to conserve volume, that
        #   is done automatically
        conserve_phase = self.params['orbit'].get('conserve','periastron')
        conserve_volume = e>0
        if conserve_volume and 'conserve' in self.params['orbit']:
            if self.params['orbit']['conserve']=='equipot':
                conserve_volume = False
        max_iter_volume = 10 if conserve_volume else 1 # number of iterations for conservation of volume (max)
        #-- we do not need to calculate bolometric luminosities if we don't include
        #   the reflection effect
        do_reflection = False
        #-- compute new mesh if this is the first time set_time is called, or
        #   if the eccentricity is nonzero
        if self.time is None or e>0:
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
                    print("t0 = {}, t_conserve = {}, {}".format(t0,crit_times[cvol_index], conserve_phase))
                    #self.compute_mesh(crit_times[cvol_index]-P/2.0,conserve_volume=True)
                    self.compute_mesh(crit_times[cvol_index],conserve_volume=True)
                    
                #-- else we still need to compute the mesh at *this* time!
                else:
                    self.compute_mesh(time,conserve_volume=True)
                #-- the following function both reprojects the surface to the
                #   value of the instantaneous potential and recomputes the
                #   value of the potential to conserve volume (if max_iter>1)
                if e>0:
                    self.conserve_volume(time,max_iter=max_iter_volume)
            #-- else, we have already computed the mesh once, and all we need
            #   to do is either just reset it, or conserve the volume at this
            #   new time point
            else:
                self.reset_mesh()
                if e>0:
                    self.conserve_volume(time,max_iter=max_iter_volume)
            #-- once we have the mesh, we need to place it into orbit
            keplerorbit.place_in_binary_orbit(self,time)
            #-- compute polar radius and logg!
            self.surface_gravity()
            self.temperature()
            self.intensity(ref=ref)
            if do_reflection:
                self.intensity(ref='__bol')
        else:
            self.reset_mesh()
            #-- once we have the mesh, we need to place it into orbit
            keplerorbit.place_in_binary_orbit(self,time)
        self.detect_eclipse_horizon(eclipse_detection='simple')
        self.time = time

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
        Phi = self.params['component'].get_value('pot')
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
        Phi = self.params['component'].get_value('pot')
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
        ld_law = 5
        ldbol_law = 5
        new_dtypes = []
        old_dtypes = self.mesh.dtype.names
        #-- check if the following required labels are in the mesh, if they
        #   are not, we'll have to add them
        required = [('ld___bol','f8',(5,)),('proj___bol','f8'),
                    ('logg','f8'),('teff','f8'),('abun','f8')]
        for req in required:
            if not req[0] in old_dtypes:
                new_dtypes.append(req)
        if 'pbdep' in self.params:
            for pbdeptype in self.params['pbdep']:
                for ipbdep in self.params['pbdep'][pbdeptype]:
                    ipbdep = self.params['pbdep'][pbdeptype][ipbdep]
                    if not 'ld_{0}'.format(ipbdep['ref']) in old_dtypes:
                        new_dtypes.append(('ld_{0}'.format(ipbdep['ref']),'f8',(5,)))
                        new_dtypes.append(('proj_{0}'.format(ipbdep['ref']),'f8'))
                        new_dtypes.append(('velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
                        new_dtypes.append(('_o_velo_{0}_'.format(ipbdep['ref']),'f8',(3,)))
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
        
        
    def conserve_volume(self,time,max_iter=10,tol=1e-6):
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
        R = self.params['component'].request_value('r_pole','Rsol')
        sma = self.params['orbit'].request_value('sma','Rsol')
        e = self.params['orbit'].get_value('ecc')
        P = self.params['orbit'].get_value('period','s')
        q = self.params['orbit'].request_value('q')
        F = self.params['component'].request_value('syncpar')
        oldpot = self.params['component'].request_value('pot')
        M1 = self.params['orbit'].get_constraint('mass1','kg') # primary mass in solar mass
        M2 = self.params['orbit'].get_constraint('mass2','kg') # secondary mass in solar mass
        component = self.get_component()+1
        theta = self.params['orbit'].get_value('theta','rad')
        T0 = self.params['orbit'].get_value('t0')
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
            R = np.linalg.norm(R)
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
            self.params['component']['pot'] = oldpot
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
        
    def set_time(self,time,ref='all'):
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
        self.intensity(ref=ref)
        if do_reflection:
            self.intensity(ref='__bol')
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
    
    def set_time(self,*args,**kwargs):
        self.reset_mesh()
        super(BinaryStar,self).set_time(*args,**kwargs)
        #keplerorbit.place_in_binary_orbit(self,*args)
        n_comp = self.get_component()
        component = ('primary','secondary')[n_comp]
        orbit = self.params['orbit']
        loc,velo,euler = keplerorbit.get_binary_orbit(self.time,orbit,component)
        self.rotate_and_translate(loc=loc,los=(0,0,+1),incremental=True)
        
    
    
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
    
    
    def projected_intensity(self,los=[0.,0.,+1],ref=0,method=None,with_partial_as_half=True):
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
                proj_intens = (1-transit.occultuniform(z,p)[0])*total_flux
             #-- assume claret limb darkening and dark component
            elif ld_func=='claret':
                logger.info("projected intensity with analytical Claret LD law")
                cn = self.mesh['ld_'+ref][0,:4]
                
                try:
                    proj_intens = transit.occultnonlin(z,p,cn)[0]*total_flux
                except ValueError:
                    proj_intens = total_flux
            elif ld_func=='quadratic':
                raise NotImplementedError
                logger.info("proj. intensity with analytical quadratic LD law")
                cn = self.mesh['ld_'+ref][0,:2]
                proj_intens = transit.occultquad(z,p,cn)[0]
            l3 = idep['l3']
            pblum = idep['pblum']
            return proj_intens*pblum + l3


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
        only_adjust = False
        
    def par_to_str(val,color=True):
        adjust = val.get_adjust()
        par_repr = val.to_str()
        N = len(par_repr)
        if adjust is True and color is True:
            par_repr = "\033[32m" + '\033[1m' + par_repr + '\033[m'
        elif adjust is False and color is True:
            par_repr = "\033[31m" + '\033[1m' + par_repr + '\033[m'
        elif adjust is True and color is False:
            par_repr = '*'+par_repr
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


if __name__=="__main__":
    import doctest
    doctest.testmod()
