"""
.. autosummary::    

   phoebe.parameters.parameters.Parameter
   
Better create a custom table:

Bla :py:class:`Bla <phoebe.parameters.parameters.Parameter>`

Bla :py:class:`Parameter`
"""

#-- make some functions available in the root
from parameters.parameters import ParameterSet,Parameter
from parameters.parameters import ParameterSet as PS
from parameters.parameters import load as load_ps
from parameters import create
from parameters.datasets import DataSet,LCDataSet,IFDataSet,SPDataSet,RVDataSet
from parameters.tools import add_rotfreqcrit
from backend.universe import Star,BinaryRocheStar,BinaryStar,BodyBag,BinaryBag
from backend.universe import load as load_body
from backend.observatory import compute,observe,add_bitmap,image,ifm
from backend.fitting import run
from utils.utils import get_basic_logger
from units import constants
from units.conversions import convert,Unit
from wd import wd