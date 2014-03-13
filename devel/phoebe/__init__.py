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

Section 2.1 Limb darkening & local intensities
-----------------------------------------------

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


See the :ref:`limbdark <limbdark-atmospheres>` module for more information.

Section 2.2 Reflection and heating
-----------------------------------

By default, reflection and heating effects are not calculated. You can switch
them on via ``refl=True`` and ``heating=True``. You can switch off the calculation
of irradiation of objects that are too dim or too cool, by setting the parameter
``irradiator=False`` in the Body parameterSet (star, component). Note that they
will still be irradiated, but other bodies will not receive their radiation.
If you don't want a body to take part in anything, you will need to separate it
from the others (e.g. in a disconnected BodyBag).

The following parameters are important in the reflection and heating effects:

    - ``alb``: highly reflective surface have a high albedo (e.g. snow is close
      to unity). Absorbing surfaces have a low albedo (e.g. a coal). A low albedo
      implies a high heating capacity :ref:`(more info) <label-alb-lcdep-phoebe>`
    - ``redist`` heat redistribution parameter. If zero, no heat will be
      redistribution. If unity, the heat redistribution is instantaneous.
      :ref:`(more info) <label-redist-star-phoebe>`
    - ``redisth``: sets the fraction of redistributed heat to be only along lines
      of constant latitude. :ref:`(more info) <label-redisth-star-phoebe>`

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


Section 2.4 Light time travel effects
--------------------------------------

Light time travel effects are off by default. You can enable them by setting
``ltt=True``. In this case, the orbit will be precomputed
:py:func:`(barycentric orbit) <phoebe.dynamics.keplerorbit.get_barycentric_orbit>`,
such that for any given barycentric time, we know what the proper time of the
components is. Note that the relative light time travel effects *between* the
components is not taken into account, i.e. reflection/heating effects are not
treated correctly.


Section 2.5 Eclipse detection
--------------------------------------

The default eclipse detection is optimized for binaries but will not work
otherwise. Depending on the circumstances, you can choose any of the following:

TBD

Section 2.6 Subdivision
--------------------------------------

TBD

Section 2.7 Interstellar reddening
--------------------------------------

TBD

Section 2.8 Pulsations
--------------------------------------

TBD

Section 2.9 Magnetic fields
--------------------------------------

TBD

Section 2.10 Spots
--------------------------------------

TBD

Section 2.11 Potential shapes
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
#-- make some important classes available in the root:
from .parameters.parameters import ParameterSet,Parameter
from .parameters.parameters import ParameterSet as PS
from .parameters.datasets import DataSet,LCDataSet,IFDataSet,SPDataSet,RVDataSet
from .backend.universe import Star,BinaryRocheStar,MisalignedBinaryRocheStar,\
                              BinaryStar,BodyBag,BinaryBag,AccretionDisk,\
                              PulsatingBinaryRocheStar
from .frontend.bundle import Bundle

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

#-- common modules
from .backend import observatory,universe,plotting,fitting
from .wd import wd

#-- common extras
from .units import constants
from .units.conversions import convert,Unit

    
