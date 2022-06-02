import numpy as np
import phoebe
phoebe.interactive_checks_off()
phoebe.interactive_constraints_off()
from phoebe.dependencies import distl

class System():
    
    def __init__(self, dataset_size, phases_size, check_eclipse = True):
        '''
        Builds a PHOEBE light curve database of dataset_size light curves 
        computed in phases_size phase points from [-0.5,0.5].
        
        Parameters
        ----------
        dataset_size: int 
            Number of light curves to compute
        phases_size: int
            Number of phases between [-0.5, 0.5] to sample the light curves in
        check_eclipse: bool
            Whether to check if each randomly generated system is eclipsing before computing it.
        '''
        self.size = dataset_size 
        self.phases_size = phases_size
        self.check_eclipse = check_eclipse
        
    def sample_dist(self):
        '''Generates samples based on the parameter distributions and dataset size'''
        self.samples = {}
        for key in self.distributions.keys():
            self.samples[key] = self.distributions[key].sample(self.size)
            
            # transform values outside of priors
            mask_low = self.samples[key] < self.priors[key].low
            mask_high = self.samples[key] > self.priors[key].high 
            self.samples[key][mask_low] = self.priors[key].low+np.abs(self.priors[key].low-self.samples[key][mask_low])
            self.samples[key][mask_high] = self.priors[key].high-np.abs(self.priors[key].high-self.samples[key][mask_high])
    
    def build_default_bundle(self):
        pass
    
    def adjust_gravb_alb(self):
        '''Automatically set the gravity darkening coeffs and albedoes in PHOEBE'''
        if self.bundle.get_value('teff', component='secondary', context='component') > 8000:
            self.bundle.set_value('gravb_bol', component='secondary', value=1.0)
            self.bundle.set_value('irrad_frac_refl_bol', component='secondary', value=1.0)
            
    def check_eclipse_condition(self):
        '''Checks if the system is eclipsing'''
        rsum = self.bundle.get_value('requivsumfrac', context='component')
        per0 = self.bundle.get_value('per0', context='component', unit='rad')
        ecc = self.bundle.get_value('ecc', context='component')
        incl = self.bundle.get_value('incl@binary', context='component', unit='rad')
        
        r_sc = (1-ecc**2)/(1+ecc*np.cos(np.pi/2 - per0))
        r_ic = (1-ecc**2)/(1+ecc*np.cos(np.pi/2 + per0))
        
        if r_sc*np.cos(incl) <= rsum or r_ic*np.cos(incl) <= rsum:
            return True
        else:
            return False
        
    def compute_model(self, point, **kwargs):
        '''Computes an individual model'''
        values = np.zeros(len(self.distributions.keys()))
        for i,key in enumerate(self.distributions.keys()):
            values[i] = self.samples[key][point]
            self.bundle.set_value(key, self.samples[key][point])
            
        if self.check_eclipse:
            eclipsing = self.check_eclipse_condition()
        else:
            eclipsing = True
        
        if eclipsing:
            try:
                self.bundle.run_checks()
                self.bundle.run_delayed_constraints()
                self.adjust_gravb_alb()

                self.bundle.run_compute(**kwargs)
                return values, self.bundle.get_value('fluxes', dataset='lc01', model='latest')

            except:
                return values, np.nan*np.ones(self.phases_size)
        else:
            return values, np.nan*np.ones(self.phases_size)
        
        
    def compute_dataset(self, lcs_file, params_file, save_rate=10, **kwargs):
        '''Computes the dataset and iteratively saves the results to a light curves and parameters files.'''
        self.sample_dist()
        self.build_default_bundle()
        lcs = np.nan*np.ones((self.size, self.phases_size))
        params = np.nan*np.ones((self.size, len(self.distributions.keys())))
        
        for i in range(self.size):
            params[i], lcs[i] = self.compute_model(i, **kwargs)
            if i%save_rate==0:
                np.save(params_file, params)
                np.save(lcs_file, lcs)
                

class Detached(System):
    
    def __init__(self, dataset_size, phases_size):
        '''Initializes a detached systems database with pre-determined parameter 
        distributions and priors.'''
        super(Detached, self).__init__(dataset_size, phases_size)
        self.distributions = {
            'incl@binary': distl.gaussian(90,30),
            'teffratio': distl.gaussian(1, 0.66),
            'requivsumfrac': distl.uniform(0.01, 0.8),
            'requivratio': distl.gaussian(1, 0.66),
            'ecc': distl.gaussian(0., 1.0),
            'per0': distl.uniform(0,360),
            'q': distl.gaussian(1.0,0.66)
            }
        
        self.priors = {
            'incl@binary': distl.uniform(0,90),
            'teffratio': distl.uniform(0.1, 1),
            'requivsumfrac': distl.uniform(0.0, 1.),
            'requivratio': distl.uniform(0, 10),
            'ecc': distl.uniform(0., 1),
            'per0': distl.uniform(0,360),
            'q': distl.uniform(0.1,10)
            }
    
    def build_default_bundle(self):
        '''Builds a default PHOEBE detached system bundle.'''
        b = phoebe.default_binary()
        b.add_constraint('requivsumfrac')
        b.add_constraint('requivratio')
        b.add_constraint('teffratio')
        
        b.flip_constraint('requivsumfrac', solve_for='requiv@primary')
        b.flip_constraint('requivratio', solve_for='requiv@secondary')
        b.flip_constraint('teffratio', solve_for='teff@secondary')
        
        b.set_value('teff@primary', 15000)
        b.set_value('gravb_bol@primary', 1.0)
        b.set_value('irrad_frac_refl_bol@primary', 1.0)
        
        b.add_dataset('lc', compute_phases = np.linspace(-0.5,0.5,self.phases_size))
        b.set_value_all('atm', 'blackbody')
        b.set_value_all('ld_mode', 'manual')
        b.set_value_all('ld_func', 'logarithmic')
        b.set_value_all('ld_coeffs', [0.5, 0.5])
        
        self.bundle = b
        
        
class Contact(System):
    
    def __init__(self, dataset_size, phases_size):
        '''Initializes a contact systems database with pre-determined parameter 
        distributions and priors.'''
        super(Contact, self).__init__(dataset_size, phases_size)
        self.distributions = {
            'incl@binary': distl.gaussian(90,50),
            'teffratio': distl.gaussian(1, 0.2),
            'fillout_factor': distl.uniform(0.05, 0.95),
            'q': distl.uniform(0.01,10)   
        }
        self.priors = {
            'incl@binary': distl.uniform(0,90),
            'teffratio': distl.uniform(0.1, 1),
            'fillout_factor': distl.uniform(0,1),
            'q': distl.uniform(0.,10)
            }
        
    def build_default_bundle(self):
        '''Builds a default PHOEBE contact system bundle.'''
        b = phoebe.default_binary(contact_binary=True)
        b.flip_constraint('pot', solve_for='requiv@primary')
        b.flip_constraint('fillout_factor', solve_for='pot')
        
        b.add_constraint('teffratio')
        b.flip_constraint('teffratio', solve_for='teff@secondary')
        
        b.set_value('teff@primary', 7000)
        
        b.add_dataset('lc', compute_phases = np.linspace(-0.5,0.5,self.phases_size))
        b.set_value_all('atm', 'blackbody')
        b.set_value_all('ld_mode', 'manual')
        b.set_value_all('ld_func*', 'logarithmic')
        b.set_value_all('ld_coeffs*', [0.5, 0.5])
        
        self.bundle = b