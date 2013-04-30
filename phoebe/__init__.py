"""

Section 1. Package structure
============================

**Main packages**: these are most commonly used to compute models and fit parameters.

.. autosummary::    

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
   
   
Section 2. Wonder how things are done?
======================================

Section 2.1 Model computations
-------------------------------

* How are surface deformations done?
    - bla
* How is gravity darkening taken into account?
    - bla
* How is limb darkening treated?


Section 2.2 Generating observables
-------------------------------------   

* How is interferometry computed?
* How are spectra computed?
* How are Stokes profiles computed?


Section 2.3 Fitting and statistics
-----------------------------------

* How is the goodness-of-fit of a model computed?   
   
   
"""
#-- make some important classes available in the root:
from .parameters.parameters import ParameterSet,Parameter
from .parameters.parameters import ParameterSet as PS
from .parameters.datasets import DataSet,LCDataSet,IFDataSet,SPDataSet,RVDataSet
from .backend.universe import Star,BinaryRocheStar,MisalignedBinaryRocheStar,BinaryStar,BodyBag,BinaryBag,AccretionDisk

#-- common input and output
from parameters.parameters import load as load_ps
from backend.universe import load as load_body
from parameters.datasets import parse_phot,parse_rv,parse_spec_as_lprof,parse_vis2
from parameters import create
from utils.utils import get_basic_logger

#-- common tasks
from parameters.tools import add_rotfreqcrit
from backend.observatory import compute,observe,add_bitmap,image,ifm
from backend.fitting import run

#-- common modules
from backend import observatory,universe,plotting,fitting
from wd import wd

#-- common extras
from units import constants
from units.conversions import convert,Unit
    