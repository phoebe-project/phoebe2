import phoebe;
import matplotlib.pyplot as plt;
import numpy as np;
from phoebe.units import constants
from phoebe.utils import coordinates
import types

       
# Define a function to dynamically add methods to a class instance           

# properttis
period			= 12.913780;
t0			= 8247.966;
dpdt			= 5.9966e-7
ecc			= 0.0;
q			= 4.484;
sma			= 58.349;
incl			= 86;
pot1			= 9.6
pot2			= 43;
teff1			= 13000;
teff2			= 30000;
rdisk			= 30.0 # 30.0
hdisk			= 8.0;
mass1			= 2.91;
mass2			= mass1*q;
rad1			= 5.97;

logger = phoebe.utils.utils.get_basic_logger(clevel='DEBUG')

# Parameter sets of components - directly components
p_donor	= phoebe.ParameterSet(context='component', teff=teff1, 
			 pot=pot1, atm='kurucz_p00', 
			 ld_func='claret', ld_coeffs='kurucz_p00',
			 label='donor', irradiator=False);
p_gainer= phoebe.ParameterSet(context='star', radius=(rad1, 'Rsol'),
			teff=teff2, mass=mass2, atm='kurucz_p00', 
			ld_func='claret', ld_coeffs='kurucz_p00',
			label='gainer', irradiator=True,
			rotperiod=period);
p_disk = phoebe.ParameterSet(context='accretion_disk', incl=(incl, 'deg'), rout=(rdisk, 'Rsol'),
			rin=(20, 'Rsol'), height=(0.1, 'Rsol'), dmdt=1e-5,
			mass=(mass2, 'Msol'), irradiator=False, redist=0.0)
#			atm='kurucz_p00', ld_func='claret', ld_coeffs='kurucz_p00');
p_orbit	= phoebe.ParameterSet(context='orbit', period=(period, 'd'),
			t0=(t0, 'd'), ecc=ecc, sma=(sma, 'Rsol'), 
			q=q, long_an=(0.0, 'deg'), 
			c1label='gainer', c2label='donor',
			label='orbit', incl=incl);
# of meshes
mesh_disk = phoebe.ParameterSet(context='mesh:disk', radial=30, longit=60);
mesh_star = phoebe.ParameterSet(context='mesh:marching');

# Bodies
gainer = phoebe.Star(p_gainer, mesh=mesh_star, label='gainer');
donor = phoebe.BinaryRocheStar(p_donor,  orbit=p_orbit, mesh=mesh_star);
disk = phoebe.AccretionDisk(p_disk, mesh=mesh_disk, label='disk');


#Place the disk and the gainer within a bodybag
primary	= phoebe.BodyBag([gainer, disk], label='primary');
p_orbit['c1label'] = 'primary';

# Bodybag with the whole system
systemBag = phoebe.BinaryBag([primary, donor], orbit=p_orbit, label='system');

# And place it within a bundle
system  = phoebe.Bundle(systemBag);

# Attach a light curve
system.lc_fromarrays(time=np.linspace(0.0,13.0,30), passband='JOHNSON.V');

# define properties of the compute
system['eclipse_alg@preview@compute']= 'full';
system['irradiation_alg@preview@compute']= 'point_source';
# turns on the heating
system['heating@preview@compute']= True;

# run the computation
system.run_compute('preview');
system.plot_syn('lc01', 'k-');
plt.savefig('bl_tr1.png');
plt.close();





