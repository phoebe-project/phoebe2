import phoebe
import numpy as np
import phoebeBackend as phb
import scipy.stats as st
import phoebe as phb2
import matplotlib.pyplot as plt
from os.path import isfile

"""
phb1_bun: Creates a bundle which mimics phoebe1 characteristics. 

filename - inpute phoebe1 file
gsize - grid size for the wdmesh
atm - atmosphere file to use. 

"""

def phb1_bun( filename='default.phoebe', atm='/home/hpablo/phoebe-code/obsolete/wd/atmcof.dat'):

	mybundle = phb2.Bundle(filename)
	mesh = phoebe.ParameterSet(context = 'mesh:wd')
	mybundle.get_object('primary').params['mesh'] = mesh
	mybundle.get_object('secondary').params['mesh'] = mesh
	mybundle._build_trunk()
	gsize1 = phb.getpar('phoebe_grid_finesize1')
#	gsize2 = phb.getpar('phoebe_grid_finesize2')
	mybundle.set_value('gridsize', gsize1)
#	mybundle.set_value('gridsize@mesh:wd@secondary', gsize2)
	mybundle['subdiv_num@legacy'] = 0

	mybundle.set_value_all('atm', atm)

	return mybundle

"""
Create grid: Create a grid of parameters that has been flattened and put into a dictionary.

par - list of parameters
pmin - list of parameter minimums
pmax - list of parameter maximums
steps - list of step sizes

returns - dictionary of flattened parameters

"""
def create_grid(par, pmin, pmax, steps):
	pars = []
	for x in range(len(par)):
		values = np.linspace(pmin[x],pmax[x], steps[x])
		pars.append(values)
		
		
	grid = np.meshgrid(*pars)
	pdict = {}
	for x in range(len(par)):
		fpar = grid[x].flatten()
		pdict[par[x]] = fpar

	return pdict
		
"""
create_lcs: Create and save lcs and parameter files for each grid element

pdict - dictionary of grid parameters and their corresponding arrays
ph - list of phases over which to make light curves
atm - atmosphere file
loc - location of output files
filename - phoebe 1 file to load
parfile - file which keeps a record of the specific parameters for a model
heating - heating added (yes or no)
reflection - reflection added (yes or no)

"""

def create_lcs(pdict, ph, atm, loc='', filename='default.phoebe', parfile='model-pars.txt', reflect ='no', heating='no'):

	f = open(parfile, 'w')
	f.write('# Model	')
	for j in range(len(pdict.keys())):
		f.write(str(pdict.keys()[j])+'	')
	f.write('\n')
	f.close()
#f.write(mrf[x]+'	'+teff1f[x]+'	'+perf[x]+'	'+logg1f[x]+'	'+logg2f[x])
	for x in range(len(pdict[pdict.keys()[0]])):
		print "MODEL "+str(x)+' of '+str(len(pdict[pdict.keys()[0]]))
# set up model parameters
		phb.open(filename)
		phb.setpar("phoebe_lcno", 1)
		
		f = open(parfile, 'a+')
		f.write(str(x)+'	')
		for i in range(len(pdict.keys())): 
			phb.setpar(pdict.keys()[i], pdict[pdict.keys()[i]][x])
			#	write model parameters		

			f.write(str(np.round(pdict[pdict.keys()[i]][x],2))+'	')
		f.write('\n')
		f.close()
#	phb.setpar("phoebe_teff1", teff1f[x])
#	phb.setpar("phoebe_period", perf[x])
#	phb.setpar("phoebe_rm", mrf[x])
	
# save to a file and load into phoebe 2
		phb.save(loc+'model'+str(x)+'.phoebe')

# set up phoebe 2 systems

	
		wdbun = phb1_bun(filename = 'model'+str(x)+'.phoebe', atm = atm)
		marbun = phb2.Bundle('model'+str(x)+'.phoebe')
		marbun.set_value_all('atm', atm)

#create times and load
	
		ph = np.linspace(-0.5, 0.5, 201)

		wdbun.lc_fromarrays(phase=ph)
		marbun.lc_fromarrays(phase=ph)

#compute preferences

		if reflect == 'yes':	 
			wdbun['refl@legacy'] = True
#		phb.setpar("phoebe_alb1", alb1)
#		phb.setpar("phoebe_alb2", alb2)
			marbun['refl@legacy'] = True
		
		else:
			wdbun['refl@legacy'] = False
#		phb.setpar("phoebe_alb1", 0.0)
#		phb.setpar("phoebe_alb2", 0.0)
			marbun['refl@legacy'] = False

		if heating == 'yes':
		
			wdbun['heating@legacy']= True
			marbun['heating@legacy']= True
		
		else:
		
			wdbun['heating@legacy']= False
			marbun['heating@legacy']= False

# subdivisions (must be zero for wd)

	
		marbun['subdiv_num@legacy'] = 8

# compute lcs

		wdbun.run_compute('legacy')
		marbun.run_compute('legacy')
	
		lc_ph1 = phb.lc(tuple(ph.tolist()), 0)
		lc_ph2 = wdbun['lc02@lcsyn'].asarray()['flux']
		lc_ph3 = marbun['lc02@lcsyn'].asarray()['flux']
		
		lc_norm1 = lc_ph1/np.median(lc_ph1)
		lc_norm2 = lc_ph2/np.median(lc_ph2)
		lc_norm3 = lc_ph3/np.median(lc_ph3)

		lcs = np.column_stack((lc_norm1, lc_norm2, lc_norm3))
		header = 'phoebe 1	phb2-wd	phb2-marching'
		savefile = loc+'model'+str(x)+'.dat'
		np.savetxt(savefile, lcs, header=header)



	return

