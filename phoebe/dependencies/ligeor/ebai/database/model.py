import numpy as np
from phoebe.dependencies.ligeor.models import polyfit, twogaussian
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt

class DatabaseModel():
    
    def __init__(self, analytical_model, phases=None, db_fluxes=None, db_params=None, db_model=None, db_model_chi2=None):
        '''Initializes a database of light curves that can be fitted with an analytical model.
        
        Parameters
        ----------
        analytical_model: str
            The analytical model fitted to the observed/synthetic fluxes. One of ['twogaussian', 'polyfit'].
        phases: array-like
            The array of orbital phases the database light curves have been observed in.
        db_fluxes: array-like
            The fluxes of all observed light curves in the database.
        db_params: array-like
            The physical parameters corresponding to each light curve.
        db_model: array-like
            (Optional) If analytical models of the database exist, they can be passed to the database model 
            instead of recomputing them.
        db_model_chi2: array-like
            (Optional) The chi2 values corresponding to the models provided in db_model
        '''
        self.analytical_model = analytical_model
        self.db_params = db_params
        self.phases = phases
        self.db_fluxes = db_fluxes
        self.db_model = db_model
        self.db_model_chi2 = db_model_chi2

    
    @staticmethod
    def add_random_noise(fluxes, noise_level=0.1):
        '''
        Adds random noise to a light curve.
        
        Parameters
        ----------
        noise_level: float
            The standard deviation of the normal distribution used to sample the added noise.
            
        Returns
        -------
        fluxes+noise 
        '''
        noise = np.random.normal(loc=0, scale=noise_level, size = fluxes.shape[0])
        return fluxes + noise, noise_level*np.ones_like(fluxes)
        
    def fit_datapoint(self, fluxes, add_noise_to_data=True, add_noise_to_model=False, 
                      noise_level_data=1e-8, noise_level_model=1e-3, diagnose=False, **kwargs): 
        '''
        Fits the chosen analytical model to a single light curve.
        
        Parameters
        ----------
        fluxes: array-like
            The light curve fluxes
        add_noise: bool
            If true, random noise will be added to the light curve.
        noise_level: float
            The standard deviation of the normal distribution used to sample the added noise.
        diagnose: float
            If true, the light curve and model will be plotted.
            
        Returns
        -------
        model
        model_chi2
        '''
        
        fluxes_orig = np.ndarray.copy(fluxes)
        if add_noise_to_data == False:
            if 'sigmas' not in kwargs.keys():
                warnings.warn('Option to add noise to data not checked and sigmas not provided. Adding default noise with stdev of 1e-8.')
            sigmas = kwargs.pop('sigmas', 1e-8*np.ones_like(fluxes))
        else:
            fluxes, sigmas = self.add_random_noise(fluxes, noise_level_data)

        if self.analytical_model == 'twogaussian':
            twog = twogaussian.TwoGaussianModel(phases=np.copy(self.phases), 
                                    fluxes=np.copy(fluxes), 
                                    sigmas=np.copy(sigmas), 
                                    )
            twog.fit()
            model = twog.model
            chi2 = np.sum((twog.model-fluxes_orig)**2/fluxes_orig**2)
        
        elif self.analytical_model == 'polyfit':
            pf = polyfit.Polyfit(phases=np.copy(self.phases), 
                            fluxes=np.copy(fluxes), 
                            sigmas=np.copy(sigmas), 
                            )
            pf.fit()
                
            model = pf.model
            chi2 = np.sum((pf.model-fluxes_orig)**2/fluxes_orig**2)

        else:
            raise ValueError("Unrecognized model {}. Choose one of ['twogaussian', 'polyfit']".format(self.analytical_model))
        
        if add_noise_to_model:
            model, sigmas_model = self.add_random_noise(model, noise_level_model)
            
        if diagnose:
            plt.errorbar(x=self.phases, y=fluxes, yerr=sigmas, fmt='.')
            plt.plot(self.phases, model)
            plt.show()
        
        return model, chi2
        
    def fit_database(self, add_noise_to_data=True, add_noise_to_model=False, noise_level_data = 1e-8, noise_level_model=0.001,
                           save=False, savefile='', continue_from_existing_model=False, continue_from_existing_file=False):
        '''
        Fits an analytical model to all light curves in the database.
        
        Parameters
        ----------
        add_noise: bool
            If true, random noise will be added to the light curve.
        noise_level: float
            The standard deviation of the normal distribution used to sample the added noise.
        save: bool
            Whether to save the database model to a file.
        savefile: str
            Path to save the database model to.
        continue_from_existing_model: bool
            If a db_model is present in the database, the fitting will continue from the last modeled light curve.
        continue_from_existing_file: bool
            If savefile is provided in such file exists, the fitting will continue from the last modeled light curve.
        
        Returns
        -------
        database_model
        database_model_chi2
        '''
        if self.db_model is not None:
            warnings.warn('Rewriting existing database model!')
        
        db_model = np.nan*np.ones(self.db_fluxes.shape)
        db_model_chi2 = np.nan*np.ones(self.db_fluxes.shape[0])
        
        if continue_from_existing_model:
            if self.db_model is not None and self.db_model_chi2 is not None:
                db_model_existing = self.db_model
                db_model_chi2_existing = self.db_model_chi2
                len_existing = len(db_model_existing)
                db_model[:len_existing] = db_model_existing
                db_model_chi2[:len_existing] = db_model_chi2_existing
                
        elif continue_from_existing_file:            
            db_model_existing = np.load(savefile+'.npy')
            db_model_chi2_existing = np.load(savefile+'.chi2.npy')
            
            len_existing = len(db_model_existing)
            db_model[:len_existing] = db_model_existing
            db_model_chi2[:len_existing] = db_model_chi2_existing
        
        else:
            len_existing = 0
            
        for i in tqdm(range(len(self.db_fluxes))):
            if i < len_existing:
                pass
            else:
                try:
                    db_model[i], db_model_chi2[i] = self.fit_datapoint(self.db_fluxes[i], 
                                                                    add_noise_to_data=add_noise_to_data, 
                                                                    add_noise_to_model=add_noise_to_model,
                                                                    noise_level_data=noise_level_data,
                                                                    noise_level_model=noise_level_model)
                except:
                    pass
            
        if save:
            if len(savefile) == 0:
                raise ValueError('Please provide a path to save the modeled database to with the savefile kwarg.')
            else:
                np.save(savefile, db_model)
                np.save(savefile+'.chi2', db_model_chi2)
        
        self.db_model = db_model
        self.db_model_chi2 = db_model_chi2
                
        return db_model, db_model_chi2