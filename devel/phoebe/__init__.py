"""

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
   
"""
#-- make some functions available in the root
from parameters.parameters import ParameterSet,Parameter
from parameters.parameters import ParameterSet as PS
from parameters.parameters import load as load_ps
from parameters import create
from parameters.datasets import DataSet,LCDataSet,IFDataSet,SPDataSet,RVDataSet
from parameters.datasets import parse_phot,parse_rv,parse_spec_as_lprof,parse_vis2
from parameters.tools import add_rotfreqcrit
from backend.universe import Star,BinaryRocheStar,BinaryStar,BodyBag,BinaryBag
from backend.universe import load as load_body
from backend.observatory import compute,observe,add_bitmap,image,ifm
from backend.fitting import run
from utils.utils import get_basic_logger
from units import constants
from units.conversions import convert,Unit
from wd import wd