import os

if os.getenv('PHOEBE_ENABLE_PLOTTING', 'TRUE').upper() == 'TRUE':
    try:
        from . import autofig
    except ImportError as e:
        print("autofig could not be imported with error: {}.  Plotting will be disabled.".format(e))

from . import nparray
from . import distl
from . import unitsiau2015
