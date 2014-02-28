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
    
    - ``atm``: sets the local intensities
    - ``ld_coeffs``: sets the limb darkening coefficients
    - ``ld_func``: sets the limb darkening function

The physically most consistent way of working is by setting ``atm`` and ``ld_coeffs``
to some grid name (e.g. kurucz, blackbody). Then local intensities and limb
darkening coefficients will be interpolated to match the local conditions
(effective temperature, gravity) on the star.

Sometimes, you want to specify a set of limb darkening coefficients yourself. In 
that case, supply a list with coefficients to ``ld_coeffs``.

.. warning::

    In Phoebe2, this set of three parameters occur many times. First of all,
    the main body itself (parameterSets component and star) contain them. These
    represent the bolometric atmospheric properties.
    
    Furthermore, these three parameters also occur in *each* passband-dependable
    parameterSet (lcdep, rvdep...). The reason for this is optimal flexibility:
    you might want to use a different atmosphere table for UV data as for
    infrared data.
    
    So to recap: you have to specify bolometric properties of the atmospheres,
    but also the passband properties. Each time you add observables in a specific
    passband, set the ``atm``, ``ld_coeffs`` and ``ld_func`` as well!

Section 2.2 Reflection and heating
-----------------------------------

By default, reflection and heating effects are not calculated. You can switch
them on via ``refl=True`` and ``heating=True``. You can switch off the calculation
of irradiation of objects that are too dim or too cool, by setting the parameter
``irradiator=False`` in the Body parameterSet (star, component). Note that they
will still be irradiated, but other bodies will not receive their radiation.

The following parameters are important in the reflection and heating effects:

    - ``alb``
    - ``redist``
    - ``redisth``


Section 2.3 Beaming
-----------------------------------

By default, beaming effects are not calculated. You can switch it on by setting
the ``beaming_alg`` accordingly (i.e. not equal to ``none``). If you decide that
the beaming contribution of a particular component is not worth it, you can set
``beaming=False`` in the Body parameterSet (star, component). There are three
options for the beaming algorithm:

    - ``beaming_alg='full'`` (slowest): local intensities and limb darkening
      coefficients are consistently computed via velocity shifts in the original
      specific intensity grids
    - ``beaming_alg='local'`` (moderate): beaming is computed by setting local
      beaming factors on the star
    - ``beaming_alg='simple'`` (fastest): beaming  is computed by setting a
      global beaming factor.

The latter two options are particularly useful for circular orbits, as the
beaming coefficients only need to be computed ones.


Section 2.4 Light time travel effects
--------------------------------------

Light time travel effects are off by default. You can enable them by setting
``ltt=True``.


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

    
