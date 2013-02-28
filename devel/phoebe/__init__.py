
#-- make some functions available in the root
from parameters.parameters import ParameterSet,Parameter
from parameters.parameters import ParameterSet as PS
from parameters.parameters import load as load_ps
from parameters.datasets import DataSet,LCDataSet,IFDataSet,SPDataSet,RVDataSet
from parameters.tools import add_rotfreqcrit
from backend.universe import Star,BinaryRocheStar,BinaryStar,BodyBag,BinaryBag
from backend.universe import load as load_body
from backend.observatory import compute,observe,add_bitmap,image,ifm
from backend.mcmc import run_mcmc
from backend.nonlin import run_nonlin
from utils.utils import get_basic_logger
from units import constants
from units.conversions import convert,Unit
from wd import wd