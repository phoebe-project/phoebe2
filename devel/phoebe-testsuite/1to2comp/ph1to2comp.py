import phoebe
import numpy as np
import phoebeBackend as phb
import scipy.stats as st
import phoebe as phb2
import matplotlib.pyplot as plt
from os.path import isfile
import compfun as cf

"""
This program creates lightcurves from phoebe2 to compare with phoebe1 in a variety of situations. Every parameter iteration will produce 3 different lcs:

1.) Phoebe 1 lcs
2.) Phoebe 2 lcs with phoebe1 meshes and atm.
3.) Phoebe 2 lcs with marching meshes and phoebe1 atm.

"""


#phoebe 1 file

filename = 'default.phoebe'

#Initialize phoebe1 lc

phb.init()
phb.configure()
phb.open(filename)
phb.setpar("phoebe_lcno", 1)

# Initial conditions for testing

#Choose your atmosphere- you must change to the location of phoebe devel branch on your system.
# blackbody 

#atm = '/home/hpablo/phoebe-code/obsolete/wd/atmcofplanck.dat'


# kurucz
atm = '/home/hpablo/phoebe-code/obsolete/wd/atmcof.dat'

phb.setpar('phoebe_atm1_switch', 1) # 0 for blackbody

#gridsize

gsize = 20
phb.setpar('phoebe_grid_coarsesize1', gsize)
phb.setpar('phoebe_grid_coarsesize2', gsize)
phb.setpar('phoebe_grid_finesize1', gsize)
phb.setpar('phoebe_grid_finesize2', gsize)

#limb darkening must set lcx1 coefficients
# these can only be ngsize = 20
phb.setpar('phoebe_grid_coarsesize1', gsize)
phb.setpar('phoebe_grid_coarsesize2', gsize)
phb.setpar('phoebe_ld_model', 'Logarithmic law')
phb.setpar('phoebe_ld_lcx1', 0.0, 0)
phb.setpar('phoebe_ld_lcx2', 0.0, 0)
phb.setpar('phoebe_ld_lcy1', 0.0, 0)
phb.setpar('phoebe_ld_lcy2', 0.0, 0)

#gravity brightening

phb.setpar('phoebe_grb1',0.0)
phb.setpar('phoebe_grb1',0.0)

#reflection
reflect = 'no'
phb.setpar('phoebe_reffect_reflections', 0)
phb.setpar('phoebe_alb1', 0.0)
phb.setpar('phoebe_alb2', 0.0)


#heating
heating = 'no'

#synchronicity

phb.setpar('phoebe_f1', 0.0)
phb.setpar('phoebe_f2', 0.0)



phb.save('default.phoebe')

#_________________________________________________________________________#

# grid parameters (parameters given in terms of phoebe1 parameters)
# NB: all values passed to phoebe1 MUST be floats (i.e. end with .0)
# set up arrays

gpars = []
pmin = []
pmax = []
steps = []

# Mass ratio
gpars.append('phoebe_rm') # phoebe 1 keyword
pmin.append(0.25)	# min
pmax.append(1.0)	# max
steps.append(4)		# number of grid steps
# Teff primary
gpars.append('phoebe_teff1') 
pmin.append(5000.0)	
pmax.append(12000.0)	
steps.append(4)	
# Period
gpars.append('phoebe_period')
pmin.append(3.0)
pmax.append(10.0)
steps.append(4)
# Inclination
gpars.append('phoebe_incl')
pmin.append(60.0)
pmax.append(90.0)
steps.append(3)


parfile = 'model_pars.txt' #file to keep model parameters defaults to current directory if a path is not specified.
ph = np.linspace(-0.5, 0.5, 201) # array of phases to calculate models at
loc = '/home/hpablo/Dropbox/phoebe-stuff/test/' # location of output files created by create_lcs. As there will be several files I suggest you create a file to put them in.

#create grid, flatten it, and place it in a dictionary so that we have just one for loop

pdict = cf.create_grid(gpars, pmin, pmax, steps)

cf.create_lcs(pdict, ph, atm, loc=loc, filename=filename, parfile=parfile, reflect=reflect, heating=heating)
