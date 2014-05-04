"""

Section 1. Package structure
============================

**Main packages**: these are most commonly used to compute models and fit parameters.

.. autosummary::    

   phoebe.frontend
   phoebe.backend
   phoebe.parameters
   phoebe.wd
   
**Utility packages**: library of functions that Phoebe relies on, but that often can be used independently too.
   
.. autosummary::

   phoebe.algorithms
   phoebe.atmospheres
   phoebe.dynamics
   phoebe.io
   phoebe.units
   phoebe.utils

Section 2. Operation details
============================

Section 2.1 Limb darkening, local intensities and absolute fluxes
------------------------------------------------------------------

Three parameters are important for setting limbdarkening options and local intensity values:
    
    - ``atm``: sets source for values of the local intensities :ref:`(more info) <label-atm-lcdep-phoebe>`
    - ``ld_coeffs``: sets (source of) the limb darkening coefficients :ref:`(more info) <label-ld_coeffs-lcdep-phoebe>`
    - ``ld_func``: sets the limb darkening function :ref:`(more info) <label-ld_func-lcdep-phoebe>`

The physically most consistent way of working is by setting ``atm`` and ``ld_coeffs``
to some grid name (e.g. kurucz, blackbody). Then local intensities and limb
darkening coefficients will be interpolated to match the local conditions
(effective temperature, gravity) on the star. Default atmosphere is blackbody,
default limb darkening law is uniform.

Sometimes, you want to specify a set of limb darkening coefficients yourself. In 
that case, supply a list with coefficients to ``ld_coeffs``.

.. warning::

    In Phoebe2, this set of three parameters occur many times. First of all,
    the main body itself (parameterSets component and star) contain them. These
    represent the *bolometric* atmospheric properties, and are mainly relevant
    for reflection/heating effects.
    
    Furthermore, these three parameters also occur in *each* passband-dependable
    parameterSet (lcdep, rvdep...). The reason for this is optimal flexibility:
    you might want to use a different atmosphere table for UV data as for
    infrared data.
    
    So to recap: you have to specify bolometric properties of the atmospheres,
    but also the passband properties. Each time you add observables in a specific
    passband, set the ``atm``, ``ld_coeffs`` and ``ld_func`` as well!


See the :ref:`limbdark <limbdark-atmospheres>` module for more information and
:py:func:`compute_grid_ld_coeffs <phoebe.atmospheres.limbdark.compute_grid_ld_coeffs>`
for more info on how to compute limb darkening tables.

See the section on :ref:`file formats <limbdark-atmospheres-fileformats>` for
more information on the file structure of limb darkening tables.

The absolute scaling of the intensity calculations is in real physical units
(erg/s/cm2/angstrom). It takes into account the distance to the target,
interstellar reddening effects and reflection/heating effects. There are two
ways of scaling the model fluxes to observed fluxes:

   - **physical scaling**: each passband dependent parameterSet (lcdep...)
     contains a ``pblum`` parameter. When ``pblum=-1``, the parameter is
     ignored. Otherwise, the computed fluxes will be normalised to the passband
     luminosity of the component it belongs to: a spherical star at unit
     distance (1 Rsol) without extra effects (reddening, reflection...) with
     ``pblum=1`` will have observed fluxes of :math:`(4\pi)^{-1}`. Extra
     effects can decrease (reddening, distance) or increase (reflection) it.
     If only the first component has a specified ``pblum``, all other fluxes
     will be scaled **relative to the primary component in the system** (which
     is the first target in the system). In WD terms, this is dubbed "coupling
     of the luminosities". If you want to decouple the luminosity computations
     from the local atmospheric quantities for a target, then set ``pblum`` for
     that target.
   - **observational scaling**: many observable parameterSet (lcobs...) have
     two parameters ``scale`` and ``offset``. When you set them to be adjustable
     but do not specify any prior, a linear fitting will be performed to match
     the observations with the model computations. This is useful for normalised
     data, or for data for which the scaling can then be interpreted as a
     distance scaling. The offset factor can then be interpreted as
     contributions from third light. You are free to fit for only one of
     ``scale`` and ``offset``, or both. Note that in this case the model fluxes
     are not altered in situ, but only rescaled when needed (e.g. when fitting
     or plotting).

Section 2.2 Reflection and heating
-----------------------------------

By default, reflection and heating effects are not calculated. You can switch
them on via ``refl=True`` and ``heating=True``. You can switch off the calculation
of irradiation of objects that are too dim or too cool, by setting the parameter
``irradiator=False`` in the Body parameterSet (star, component). Note that they
will still be irradiated, but other bodies will not receive their radiation.
If you don't want a body to take part in anything, you will need to separate it
from the others (e.g. in a disconnected BodyBag).

The following algorithms can be chosen to compute irradiation:

    - ``irradiation_alg='point_source'`` (fast): treat the irradiating body as a
      point source. This is generally fine if the separation between the objects
      is large enough.
    - ``irradiation_alg='full'`` (slow): treat the irradiating body as an
      extended body. Physically most correct.

The following parameters are important in the reflection and heating effects:

    - ``alb``: highly reflective surface have a high albedo (e.g. snow is close
      to unity). Absorbing surfaces have a low albedo (e.g. coal). A low albedo
      implies a high heating capacity :ref:`(more info) <label-alb-lcdep-phoebe>`
    - ``redist`` heat redistribution parameter. If zero, no heat will be
      redistribution. If unity, the heat redistribution is instantaneous.
      :ref:`(more info) <label-redist-star-phoebe>`
    - ``redisth``: sets the fraction of redistributed heat to be only along lines
      of constant latitude. :ref:`(more info) <label-redisth-star-phoebe>`

With the reflection effect, the ``scattering`` phase function can additionally be
specified. The current options are:

    - ``isotropic``: Thompson scattering, e.g. in atmospheres of hot stars
    - ``henyey``: Henyey-Greenstein scattering, for either front or backwards scattering
    - ``henyey2``: Two-term Henyey-Greenstein scattering (Jupiter-like)
    - ``rayleigh``: back-front symmetric scattering
    - ``hapke``: for regolith surfaces, or even vegetation, snow...

.. warning::

    Just as for the atmospheric parameters, Phoebe2 distinguishes *bolometric*
    (part of star, component...) and *passband* albedos (part of lcdep, rvdep...).
    The bolometric albedo is used only to govern heating effects: with a bolometric
    albedo of 0.25, 75% of the incoming light will be used to heat the object,
    the remaining 25% will be reflected. The distribution over different passbands
    is set by the *passband* albedo. Note that the passband albedo sets the
    fraction of reflected versus incoming light *within the passband*. Thus,
    grey scattering means all the passband albedos are equal. A passband albedo
    of unity means no redistribution of wavelengths. A passband albedo exceeding
    unity means that light from other wavelengths is redistributed inside the
    current passband.
    
    Bolometric albedos need to be between 0 and 1, passband albedos can exceed 1.

See the :ref:`reflection <reflection-algorithms>` module for more information.

Section 2.3 Beaming
-----------------------------------

By default, beaming effects are not calculated. You can switch it on by setting
the ``beaming_alg`` accordingly (i.e. not equal to ``none``). If you decide that
the beaming contribution of a particular component is not worth it, you can set
``beaming=False`` in the Body parameterSet (star, component). There are four
options for the beaming algorithm:

    - ``beaming_alg='none'`` (immediate): no beaming is computed.
    - ``beaming_alg='full'`` (slowest): local intensities and limb darkening
      coefficients are consistently computed via velocity shifts in the original
      specific intensity grids
    - ``beaming_alg='local'`` (moderate): beaming is computed by setting local
      beaming factors on the star
    - ``beaming_alg='simple'`` (fast): beaming  is computed by setting a
      global beaming factor.

The latter two options are particularly useful for circular orbits, as the
beaming coefficients only need to be computed once.

Section 2.4 Morphological constraints
--------------------------------------

Morphological constraints are currently only implemented for binaries. There is
a parameter :ref:`morphology <label-morphology-component-phoebe>` in the
component ParameterSet for this purpose. It can take the following values:
    
    - ``morphology='unconstrained'``: has no effect
    - ``morphology='detached'``: the component must have a potential value above
      the critical value
    - ``morphology='semi-detached'``: the component must have a potential value
      equal to the critical value
    - ``morphology='overcontact'``: the component must have a potential value
      below the critical value

Note that these values are not strictly enforced, and in principle do not
interfere with any of the computations. However, a preprocessing step is added
using :py:func:`binary_morphology <phoebe.backend.processing.binary_morphology>`,
which adjusts the limits on the potential value parameter. Then it can be
easily checked if the potential value satisfies the constraint, e.g. during
fitting.


Section 2.5 Light time travel effects
--------------------------------------

Light time travel effects are off by default. You can enable them by setting
``ltt=True``. In this case, the orbit will be precomputed
:py:func:`(barycentric orbit) <phoebe.dynamics.keplerorbit.get_barycentric_orbit>`,
such that for any given barycentric time, we know what the proper time of the
components is. Note that the relative light time travel effects *between* the
components is not taken into account, i.e. reflection/heating effects are not
treated correctly.


Section 2.6 Eclipse detection
--------------------------------------

The eclipse detection algorithm is set via the parameter ``eclipse_alg``.
Depending on the circumstances, you can choose any of the following:

    - ``only_horizon``: simples, and assumes different bodies do not overlap
      at all. If the normal of a triangle is pointed in the observer's direction,
      it is assumed to be visible.
    - ``full``: checks if any triangle is eclipsed by any other. Can also detect
      self-eclipsing parts. Very slow.
    - ``graham``: Assumes bodies have convex shapes, and uses a Graham scan and
      fast "point in hull" algorithms to label eclipsed triangles.
    - ``binary`` (default): uses Graham scan algorithm (assuming convex bodies)
      to compute eclipses during expected eclipse times, and uses ``only_horizon``
      otherwise.

Section 2.7 Oversampling in time
--------------------------------------

If the exposure time of your observations is relatively long compared to the rate at which you
expect changes in the them to happen, you should oversample the computations in time. For example, 
if you want to compute a light curve taken at 60s cadence of a star that has an eclipse that lasts
only a few minutes, the ingress and egress of the eclipse will not correctly be modelled if you do
not oversample in time. 

Oversampling can be set using the parameters ``exptime`` (exposure time in seconds) and ``samprate``
(number of time points to be computed within one exposure). Both can be single values that will be
used for every time point, or arrays with the same length as the time array of the observations.
The latter can be useful to, e.g., use a higher oversampling during eclipses when the changes in 
a light curve are more rapid.

The oversampling is done as follows: say that you want to compute observables at time 100s, using an
integration time of 200s and with an oversampling of 4, then the observation started at 0s and ended
 at 200s. As a result, you get the average value of the observables at times 25s, 75s, 125s and 
 175s.

The functions in which the times extract times and refs and binoversampling are done under the hood
are :py:func:`(extract_times_and_refs) <phoebe.backend.observatory.extract_times_and_refs>` and
:py:func:`(bin_oversampling) <phoebe.backend.universe.Body.bin_oversampling>`.



Section 2.8 Subdivision
--------------------------------------

TBD


Section 2.9 Interstellar reddening
--------------------------------------

TBD

Section 2.10 Pulsations
--------------------------------------

TBD

Section 2.11 Magnetic fields
--------------------------------------

TBD

Section 2.12 Spots
--------------------------------------

TBD

Section 2.13 Potential shapes
--------------------------------------

TBD
   
Section 3. Wonder how things are done?
======================================

Section 3.1 Model computations
-------------------------------

* How are surface deformations done?
    - bla
* How is gravity darkening taken into account?
    - bla
* :ref:`How is limb darkening treated? <wonder_how_atmospheres>`


Section 3.2 Generating observables
-------------------------------------   

* How is interferometry computed?
* How are spectra computed?
* How are Stokes profiles computed?


Section 3.3 Fitting and statistics
-----------------------------------

* How is the goodness-of-fit of a model computed?   
   
   
"""

import os

# People shouldn't import Phoebe from the installation directory (inspired upon
# pymc warning message).
if os.getcwd().find(os.path.abspath(os.path.split(os.path.split(__file__)[0])[0]))>-1:
    # We have a clash of package name with the standard library: we implement an
    # "io" module and also they do. This means that you can import Phoebe from its
    # main source tree; then there is no difference between io from here and io
    # from the standard library. Thus, if the user loads the package from here
    # it will never work. Instead of letting Python raise the io clash (which
    # is uniformative to the unexperienced user), we raise the importError here
    # with a helpful error message
    if os.getcwd() == os.path.abspath(os.path.dirname(__file__)):
        raise ImportError('\n\tYou cannot import Phoebe from inside its main source tree.\n')
    # Anywhere else the source tree, it should be possible to import Phoebe.
    # However, it is still not advised to do that.
    else:
        print('\n\tWarning: you are importing Phoebe from inside its source tree.\n')
    

from _version import __version__
#-- make some important classes available in the root:
from .parameters.parameters import ParameterSet,Parameter
from .parameters.parameters import ParameterSet as PS
from .parameters.datasets import DataSet,LCDataSet,IFDataSet,SPDataSet,RVDataSet
from .backend.universe import Star,BinaryRocheStar,MisalignedBinaryRocheStar,\
                              BinaryStar,BodyBag,BinaryBag,AccretionDisk,\
                              PulsatingBinaryRocheStar
from .frontend.bundle import Bundle, load, info
from .frontend.common import take_orbit_from

#-- common input and output
from .parameters.parameters import load as load_ps
from .backend.universe import load as load_body
from .backend import office
from .parameters.datasets import parse_lc,parse_phot,parse_rv,\
                                 parse_etv,parse_spec_as_lprof,parse_vis2
from .parameters import create
from .utils.utils import get_basic_logger
from phoebe.io import parsers

#-- common tasks
from .parameters.tools import add_rotfreqcrit
from .backend.observatory import compute,observe,add_bitmap,image,ifm
from .backend.fitting import run
from .atmospheres.limbdark import download_atm

#-- common modules
from .backend import observatory,universe,plotting,fitting
from .wd import wd

#-- common extras
from .units import constants
from .units.conversions import convert,Unit

