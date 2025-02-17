import numpy as np
from scipy.optimize import curve_fit
from phoebe.dependencies.ligeor.utils.lcutils import *
from phoebe.dependencies.ligeor.models import Model


class TwoGaussianModel(Model):

    def __init__(self, phases=[], fluxes=[], sigmas=[], filename='', 
                n_downsample=1, usecols=(0,1,2), delimiter=',', phase_folded=True, period=1, t0=0):
        '''
        Computes the two-Gaussian model light curves of the input data.

        Parameters
        ----------
        phases: array-like
            Input orbital phases, must be on the range [-0.5,0.5]
        fluxes: array-like
            Input fluxes
        sigmas: array-like
            Input sigmas (corresponding flux uncertainities)
        filename: str
            Filename from which to load a PHASE FOLDED light curve.
        n_downsample: int
            Number of data points to skip in loaded light curve (for downsampling)
        usecols: array-like, len 2 or 3
            Indices of the phases, fluxes and sigmas columns in file.
        '''

        super(TwoGaussianModel, self).__init__(phases, fluxes, sigmas, filename, n_downsample, usecols, delimiter, phase_folded, period, t0)

        # this is just all the parameter names for each model
        self.twogfuncs = {'C': TwoGaussianModel.const, 
                    'CE': TwoGaussianModel.ce, 
                    'CG': TwoGaussianModel.cg, 
                    'CGE': TwoGaussianModel.cge, 
                    'CG12': TwoGaussianModel.cg12, 
                    'CG12E1': TwoGaussianModel.cg12e1,
                    'CG12E2': TwoGaussianModel.cg12e2
                    }

        self.params = {'C': ['C'],
                'CE': ['C', 'Aell', 'phi0'],
                'CG': ['C', 'mu1', 'd1', 'sigma1'],
                'CGE': ['C', 'mu1', 'd1', 'sigma1', 'Aell', 'phi0'],
                'CG12': ['C', 'mu1', 'd1', 'sigma1', 'mu2', 'd2', 'sigma2'],
                'CG12E1': ['C', 'mu1', 'd1', 'sigma1', 'mu2', 'd2', 'sigma2', 'Aell'],
                'CG12E2': ['C', 'mu1', 'd1', 'sigma1', 'mu2', 'd2', 'sigma2', 'Aell']
                }

    
    def isolate_param_vals_for_model(self, model_params, func):
        '''
        Isolates only parameter values pertaining to a model "func",
        assuming model params contains the union of all possible parameters.
        
        Parameters
        ----------
        model_params: dict
            Dictionary containing all possible parameters and their values (some can be nan!)
            Keys: 'C', 'mu1', 'd1', 'sigma1', 'mu2', 'd2', 'sigma2', 'Aell', 'phi0'
            
        Returns
        -------
        param_vals: array-like
            A subset of the parameter values required to compute "func".
        '''
        param_vals = np.zeros(len(self.params[func]))
        for i,key in enumerate(self.params[func]):
            param_vals[i] = model_params[key]
            
        return param_vals


    def fit(self, fit_funcs=[], param_vals=[]):
        '''
        Computes all two-gaussian models and chooses the best fit.

        The fitting is carried out in the following steps: 
            1. fit each of the 7 models,
            2. compute their model light curves
            3. compute each model's BIC value
            4. assign the model with highest BIC as the 'best fit'
        '''
        self.fits = {}
        self.best_fit = {}
        
        if len(fit_funcs) != 0:
            # the best fit function has been provided by the user

            if len(param_vals) == 0:
                # we need to fit just that one
                self.fit_twoGaussian_models(funcs=fit_funcs)
            else:
                for i,key in enumerate(fit_funcs):
                    self.fits[key] = param_vals[i]
 
        else:
            self.fit_twoGaussian_models()
            # compute all model light curves

        self.compute_twoGaussian_models()
        # compute corresponding BIC values
        self.compute_twoGaussian_models_BIC()
        
        # best_fit_func = self.check_fit()
        # choose the best fit as the one with highest BIC
        best_fit_func = list(self.models.keys())[np.nanargmax(list(self.bics.values()))]
        self.best_fit['func'] = best_fit_func
        self.best_fit['param_vals'] = self.fits[best_fit_func]
        self.best_fit['param_names'] = self.params[best_fit_func]
            
        self.model = self.models[best_fit_func]


    def compute_model(self, phases, best_fit=True, func='', param_values = []):
        '''
        Compute a model in an input phase array.

        Parameters
        ----------
        phases: array-like
            The input phases to compute the model in.
        best_fit: bool
            Whether to take the best fit values or user-provided ones.
        func: str
            (Optional) twoGaussian model function string to use, if best_fit is False.
        param_values: array-like
            (Optional) Parameter values of the twoGaussian model function, if best_fit is False.
        '''
        if best_fit:
            func = self.best_fit['func']
            param_values = self.best_fit['param_vals']
            
        return self.twogfuncs[func](phases, *param_values)

    def save_model(self, nbins=1000, func='', param_values = [], save_file=''):
        '''
        Save the best fit model to a file.

        Parameters
        ----------
        nbins: int
            The number of phase points between -0.5 and 0.5 to compute the model in
        save_file: str
            The filename to save to.
        '''
        
        if len(save_file) == 0:
            save_file = self.filename + '.2g'
        phases = np.linspace(-0.5,0.5,nbins)
        fluxes = self.twogfuncs[func](phases, *param_values)
        np.savetxt(save_file, np.array([phases, fluxes]).T)

    # HELPER FUNCTIONS

    @staticmethod
    def ellipsoidal(phi, Aell, phi0):
        r'''
        Ellipsoidal model, defined as $y = (1/2) A_{ell} \cos (4 \pi (\phi - \phi_0))$
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        Aell: float
            Amplitude of the elliposoidal
        phi0: float
            Phase-point to center the elliposoidal on (position of primary or secondary eclipse)

        Returns
        -------
        y: array-like
            $y = (1/2) A_{ell} \cos (4 \pi (\phi - \phi_0))$
        '''
        # just the cosine component with the amplitude and phase offset as defined in Mowlavi (2017)
        return 0.5*Aell*np.cos(4*np.pi*(phi-phi0))

    @staticmethod
    def gaussian(phi, mu, d, sigma):
        r'''
        Gaussian model, defined as $y = d \exp(-(\phi-\mu)^2/(2\sigma^2))$

        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        mu: float
            Position of the Gaussian
        d: float
            Amplitude of the Gaussian
        sigma: float
            Scale (FWHM) of the Gaussian

        Returns
        -------
        y: array-like
            $y = d \exp(-(\phi-\mu)^2/(2\sigma^2))$
        '''

        # one Gaussian
        return d*np.exp(-(phi-mu)**2/(2*sigma**2))

    @staticmethod
    def gsum(phi, mu, d, sigma):
        '''
        Copies the Gaussian to mu +/- 1 and mu +/- 2 to account for 
        extended phase-folded light curves that cover multiple orbits.
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        mu: float
            Position of the Gaussian
        d: float
            Amplitude of the Gaussian
        sigma: float
            Scale (FWHM) of the Gaussian

        Returns
        -------
        y: array-like
            Array with gaussians at mu, mu +/-1 and mu +/- 2.
        '''

        gauss_sum = np.zeros(len(phi))
        for i in range(-2,3,1):
            gauss_sum += TwoGaussianModel.gaussian(phi,mu+i,d,sigma)
        return gauss_sum

    # MODELS as defined in Mowalvi (2017)

    @staticmethod
    def const(phi, C):
        '''The constant model y = C'
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant

        Returns
        -------
        y: array-like
            $y = C$
        '''

        # constant term
        return C*np.ones(len(phi))

    @staticmethod
    def ce(phi, C, Aell, phi0):
        '''Constant + ellipsoidal model
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant
        Aell: float
            Amplitude of the elliposoidal
        phi0: float
            Phase-point to center the elliposoidal on (position of primary or secondary eclipse)

        Returns
        -------
        y: array-like
            y = const(phi, C) - ellipsoidal(phi, Aell, phi0)
        '''

        return TwoGaussianModel.const(phi, C) - TwoGaussianModel.ellipsoidal(phi, Aell, phi0)

    @staticmethod
    def cg(phi, C, mu, d,  sigma):
        '''Constant + Gaussian model
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant
        mu: float
            Position of the Gaussian
        d: float
            Amplitude of the Gaussian
        sigma: float
            Scale (FWHM) of the Gaussian

        Returns
        -------
        y: array-like
            y = const(phi, C) - gsum(phi, mu, d, sigma)
        '''
        # constant + one gaussian (just primary eclipse)
        return TwoGaussianModel.const(phi, C) - TwoGaussianModel.gsum(phi, mu, d, sigma)

    @staticmethod
    def cge(phi, C, mu, d, sigma, Aell, phi0):
        '''
        Constant + Gaussian + ellipsoidal model

        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant
        mu: float
            Position of the Gaussian
        d: float
            Amplitude of the Gaussian
        sigma: float
            Scale (FWHM) of the Gaussian
        Aell: float
            Amplitude of the elliposoidal
        phi0: float
            Phase-point to center the elliposoidal on (position of primary or secondary eclipse)

        Returns
        -------
        y: array-like
            y = const(phi, C) - gsum(phi, mu, d, sigma) - ellipsoidal(phi, Aell, mu)
        '''
        return TwoGaussianModel.const(phi, C) - TwoGaussianModel.ellipsoidal(phi, Aell, phi0) - TwoGaussianModel.gsum(phi, mu, d, sigma)

    @staticmethod
    def cg12(phi, C, mu1, d1, sigma1, mu2, d2, sigma2):
        '''
        Constant + two Gaussians model
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant
        mu1: float
            Position of the first Gaussian
        d1: float
            Amplitude of the first Gaussian
        sigma1: float
            Scale (FWHM) of the first Gaussian
        mu2: float
            Position of the second Gaussian
        d2: float
            Amplitude of the second Gaussian
        sigma2: float
            Scale (FWHM) of the second Gaussian

        Returns
        -------
        y: array-like
            y = const(phi, C) - gsum(phi, mu1, d1, sigma1) - gsum(phi, mu2, d2, sigma2) 
        '''
        return TwoGaussianModel.const(phi, C) - TwoGaussianModel.gsum(phi, mu1, d1, sigma1) - TwoGaussianModel.gsum(phi, mu2, d2, sigma2)

    @staticmethod
    def cg12e1(phi, C, mu1, d1, sigma1, mu2, d2, sigma2, Aell):
        '''
        Constant + two Gaussians + ellipsoidal centered on the primary eclipse
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant
        mu1: float
            Position of the first Gaussian
        d1: float
            Amplitude of the first Gaussian
        sigma1: float
            Scale (FWHM) of the first Gaussian
        mu2: float
            Position of the second Gaussian
        d2: float
            Amplitude of the second Gaussian
        sigma2: float
            Scale (FWHM) of the second Gaussian
        Aell: float
            Amplitude of the elliposoidal

        Returns
        -------
        y: array-like
            y = const(phi, C) - gsum(phi, mu1, d1, sigma1) - gsum(phi, mu2, d2, sigma2) - ellipsoidal(phi, Aell, mu1)
        '''
        return TwoGaussianModel.const(phi, C) - TwoGaussianModel.gsum(phi, mu1, d1, sigma1) - TwoGaussianModel.gsum(phi, mu2, d2, sigma2) - TwoGaussianModel.ellipsoidal(phi, Aell, mu1)

    @staticmethod
    def cg12e2(phi, C, mu1, d1, sigma1, mu2, d2, sigma2, Aell):
        '''
        Constant + two Gaussians + ellipsoidal centered on the secondary eclipse
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant
        mu1: float
            Position of the first Gaussian
        d1: float
            Amplitude of the first Gaussian
        sigma1: float
            Scale (FWHM) of the first Gaussian
        mu2: float
            Position of the second Gaussian
        d2: float
            Amplitude of the second Gaussian
        sigma2: float
            Scale (FWHM) of the second Gaussian
        Aell: float
            Amplitude of the elliposoidal
        phi0: float
            Phase-point to center the elliposoidal on (position of primary or secondary eclipse)


        Returns
        -------
        y: array-like
            y = const(phi, C) - gsum(phi, mu1, d1, sigma1) - gsum(phi, mu2, d2, sigma2) - ellipsoidal(phi, Aell, mu2)
        '''
        return TwoGaussianModel.const(phi, C) - TwoGaussianModel.gsum(phi, mu1, d1, sigma1) - TwoGaussianModel.gsum(phi, mu2, d2, sigma2) - TwoGaussianModel.ellipsoidal(phi, Aell, mu2)


    @staticmethod
    def lnlike(y, yerr, ymodel):
        r'''
        Computes the loglikelihood of a model.

        $\log\mathrm{like} = \sum_i \log(\sqrt{2\pi} \sigma_i) + (y_i - model_i)^2/(2\sigma_i^2)
        '''
        if yerr is not None:
            return -np.sum(np.log((2*np.pi)**0.5*yerr)+(y-ymodel)**2/(2*yerr**2))
        else:
            return -np.sum((y-ymodel)**2)

    def bic(self, ymodel, nparams):
        r'''
        Computes the Bayesian Information Criterion (BIC) value of a model.

        BIC = 2 lnlike - n_params \log(n_data)
        '''
        if self.sigmas is not None:
            return 2*self.lnlike(self.fluxes, self.sigmas, ymodel) - nparams*np.log(len(self.fluxes))
        else:
            return self.lnlike(self.fluxes, self.sigmas, ymodel)

    @staticmethod
    def find_eclipse(phases, fluxes):
        '''
        Determines the position and estimated width of the deepest eclipse.

        Parameters
        ----------
        phases: array-like
            Input phases of the light curve
        fluxes: array-like
            Input fluxes of the light curve

        Returns
        -------
        phase_min: float
            Position of the deepest eclipse in phase space
        edge_left: float
            Left edge of the eclipse in phase space
        edge_right: float
            Right edge of the eclipse in phase space
        '''

        phase_min = phases[np.nanargmin(fluxes)]
        ph_cross = phases[fluxes - np.nanmedian(fluxes) > 0]
        # this part looks really complicated but it really only accounts for eclipses split
        # between the edges of the phase range - if a left/right edge is not found, we look for
        # it in the phases on the other end of the range
        # we then mirror the value back on the side of the eclipse position for easier width computation
        try:
            arg_edge_left = np.argmin(np.abs(phase_min - ph_cross[ph_cross<phase_min]))
            edge_left = ph_cross[ph_cross<phase_min][arg_edge_left]
        except:
            arg_edge_left = np.argmin(np.abs((phase_min+1)-ph_cross[ph_cross<(phase_min+1)]))
            edge_left = ph_cross[ph_cross<(phase_min+1)][arg_edge_left]-1
        try:
            arg_edge_right = np.argmin(np.abs(phase_min-ph_cross[ph_cross>phase_min]))
            edge_right = ph_cross[ph_cross>phase_min][arg_edge_right]
        except:
            arg_edge_right = np.argmin(np.abs((phase_min-1)-ph_cross[ph_cross>(phase_min-1)]))
            edge_right = ph_cross[ph_cross>(phase_min-1)][arg_edge_right]+1

        return phase_min, edge_left, edge_right

    @staticmethod
    def estimate_eclipse_positions_widths(phases, fluxes, fix_primary=False, fix_secondary=False, diagnose_init=False):
        '''
        Determines the positions and widths of the eclipses in a light curve.

        Parameters
        ----------
        phases: array-like
            Input phases of the light curve
        fluxes: array-like
            Input fluxes of the light curve
        diagnose_init: bool
            If True, will plot the light curve with the position and width estimates.

        Returns
        -------
        estimates: dict
            A dictionary with keys 'ecl_positions' and 'ecl_widths' corresponding
            to lists of len 2 of the initial estimated values for each eclipse.
        '''
        
        if fix_primary:
            pos1 = phases[np.argmin(fluxes)]
            pos2 = pos1+0.5
            pos2 = pos2-1 if pos2 > 0.5 else pos2 
            return {'ecl_positions': [pos1, pos2], 'ecl_widths': [0.001, 0.001]}


        elif fix_secondary:
            pos1, edge1l, edge1r = TwoGaussianModel.find_eclipse(phases, fluxes)
            pos2 = pos1+0.5
            pos2 = pos2-1 if pos2 > 0.5 else pos2 
            return {'ecl_positions': [pos1, pos2], 'ecl_widths': [edge1r-edge1l, 0.005]}

        else:
            pos1, edge1l, edge1r = TwoGaussianModel.find_eclipse(phases, fluxes)
            fluxes_sec = fluxes.copy()
            fluxes_sec[((phases > edge1l) & (phases < edge1r)) | ((phases > edge1l+1) | (phases < edge1r-1))] = np.nan
            pos2, edge2l, edge2r = TwoGaussianModel.find_eclipse(phases, fluxes_sec)

            if diagnose_init:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10,8))
                plt.plot(phases, fluxes, '.')
                plt.axhline(y=np.median(fluxes), c='orange')
                for i,x in enumerate([pos1, edge1l, edge1r]):
                    ls = '-' if i==0 else '--'
                    plt.axvline(x=x, c='r', ls=ls)
                for i,x in enumerate([pos2, edge2l, edge2r]):
                    ls = '-' if i==0 else '--'
                    plt.axvline(x=x, c='g', ls=ls)
                plt.show()

            # return {'ecl_positions': [pos1, pos2], 'ecl_widths': [edge1r-edge1l, edge2r-edge2l]}
            return {'ecl_positions': [pos1, pos2], 'ecl_widths': [0.005, 0.005]}
        
    def fit_twoGaussian_models(self, init_pos=[], init_widths=[], 
                               funcs=['C', 'CE', 'CG', 'CGE', 'CG12', 'CG12E1', 'CG12E2']):
        '''
        Fits all seven models to the input light curve.
        '''
        # setup the initial parameters

        C0 = self.fluxes.max()
        ecl_dict = self.estimate_eclipse_positions_widths(self.phases, self.fluxes, diagnose_init=False)
        if len(init_pos) == 0 or len(init_widths) == 0:
            self.init_positions, self.init_widths = ecl_dict['ecl_positions'], ecl_dict['ecl_widths']
            mu10, mu20 = self.init_positions
            sigma10, sigma20 = self.init_widths
        else:
            mu10, mu20 = init_pos
            sigma10, sigma20 = init_widths

        # check if the initial estimates for the eclipses are overlapping
        if np.abs(mu20-mu10) < 5.6*sigma10 or np.abs(mu20-mu10) < 5.6*sigma20:
            mu20 = mu10 + 0.5
            mu20 = mu20 - 1 if mu20 > 0.5 else mu20

        d10 = self.fluxes.max()-self.fluxes[np.argmin(np.abs(self.phases-mu10))]
        d20 = self.fluxes.max()-self.fluxes[np.argmin(np.abs(self.phases-mu20))]
        Aell0 = 0.001
        phi0 = 0.25

        init_params = {'C': [C0,],
            'CE': [C0, Aell0, phi0],
            'CG': [C0, mu10, d10, sigma10],
            'CGE': [C0, mu10, d10, sigma10, Aell0, phi0],
            'CG12': [C0, mu10, d10, sigma10, mu20, d20, sigma20],
            'CG12E1': [C0, mu10, d10, sigma10, mu20, d20, sigma20, Aell0],
            'CG12E2': [C0, mu10, d10, sigma10, mu20, d20, sigma20, Aell0]}

        # parameters used frequently for bounds
        fmax = self.fluxes.max()
        fmin = self.fluxes.min()
        fdiff = fmax - fmin

        bounds = {'C': ((0),(fmax)),
            'CE': ((0, 1e-6, -0.5),(fmax, fdiff, 0.5)),
            'CG': ((0., -0.5, 0., 0.), (fmax, 0.5, fdiff, 0.5)),
            'CGE': ((0., -0.5, 0., 0., 1e-6, -0.5),(fmax, 0.5, fdiff, 0.5, fdiff, 0.5)),
            'CG12': ((0.,-0.5, 0., 0., -0.5, 0., 0.),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5)),
            'CG12E1': ((0.,-0.5, 0., 0., -0.5, 0., 0., 1e-6),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5, fdiff)),
            'CG12E2': ((0.,-0.5, 0., 0., -0.5, 0., 0., 1e-6),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5, fdiff))}


        # extend light curve on phase range [-1,1]
        phases_ext, fluxes_ext, sigmas_ext = extend_phasefolded_lc(self.phases, self.fluxes, self.sigmas)

        for key in funcs:
            try:
                self.fits[key] = curve_fit(self.twogfuncs[key], phases_ext, fluxes_ext, p0=init_params[key], sigma=sigmas_ext, bounds=bounds[key])[0]
            except Exception as err:
                print("2G model {} failed with error: {}".format(key, err))
                self.fits[key] = np.array([np.nan*np.ones(len(init_params[key]))])


    def compute_twoGaussian_models(self):
        '''
        Computes the model light curves given the fit solutions.
        '''
        models = {}

        for fkey in self.fits.keys():
            models[fkey] = self.twogfuncs[fkey](self.phases, *self.fits[fkey])

        self.models = models


    def compute_twoGaussian_models_BIC(self):
        '''
        Computes the BIC value of each model light curve.
        '''
        bics = {}
        nparams = {'C':1, 'CE':3, 'CG':4, 'CGE':6, 'CG12':7, 'CG12E1':8, 'CG12E2':8}

        for mkey in self.models.keys():
            bics[mkey] = self.bic(self.models[mkey], nparams[mkey])

        self.bics = bics


    def compute_eclipse_params(self, interactive=False):
        '''
        Compute the positions, widths and depths of the eclipses 
        based on the two-Gaussian model solution.

        The eclipse parameters are computed following the prescription
        in Mowlavi et al. (2017):
        - eclipse positions are set to the Gaussian positions
        - eclipse widths are 5.6 times the Gaussian FWHMs
        - eclipse depths are the constant factor minus the value of 
        the model at the eclipse positions

        Parameters
        ----------
        interactive: boolean
            If True, allows the user to manually adjust the positions of 
            eclipse minimum and edges. Default: False.

        Returns
        -------
        results: dict
            A dictionary of the eclipse paramter values.
        '''

        param_vals = self.best_fit['param_vals']
        param_names = self.best_fit['param_names']

        # gather values from the best fit solution
        sigma1 = param_vals[param_names.index('sigma1')] if 'sigma1' in param_names else np.nan
        sigma2 = param_vals[param_names.index('sigma2')] if 'sigma2' in param_names else np.nan
        mu1 = param_vals[param_names.index('mu1')] if 'mu1' in param_names else np.nan
        mu2 = param_vals[param_names.index('mu2')] if 'mu2' in param_names else np.nan
        C = param_vals[param_names.index('C')]

        # compute and adjust all available parameters, otherwise the entire eclipse is nan
        if not np.isnan(mu1) and not np.isnan(sigma1) and np.abs(sigma1) < 0.5:
            pos1 = mu1
            width1 = 5.6*np.abs(sigma1)
            depth1 = C - self.twogfuncs[self.best_fit['func']]([pos1], *self.best_fit['param_vals'])[0]
        else:
            pos1 = np.nan
            width1 = np.nan
            depth1 = np.nan
        if not np.isnan(mu2) and not np.isnan(sigma2) and np.abs(sigma2) < 0.5:
            pos2 = mu2
            width2 = 5.6*np.abs(sigma2)
            depth2 = C - self.twogfuncs[self.best_fit['func']]([pos2], *self.best_fit['param_vals'])[0]
        else:
            pos2 = np.nan
            width2 = np.nan
            depth2 = np.nan


        # compute the eclipse edges using the positons and widths
        eclipse_edges = [pos1 - 0.5*width1, pos1+0.5*width1, pos2-0.5*width2, pos2+0.5*width2]

        # # check if the resulting eclipses are overlapping:
        # if np.abs(pos2-pos1) < width1 or np.abs(pos2-pos1) < width2:
        #     # keep only the deeper eclipse
        #     if depth1 > depth2:
        #         pos1, pos2 = pos1, np.nan 
        #         width1, width2 = width1, np.nan 
        #         depth1, depth2 = depth1, np.nan
        #         eclipse_edges = [eclipse_edges[0], eclipse_edges[1], np.nan, np.nan]
        #     else:
        #         pos1, pos2 = pos2, np.nan 
        #         width1, width2 = width2, np.nan 
        #         depth1, depth2 = depth2, np.nan
        #         eclipse_edges = [eclipse_edges[2], eclipse_edges[3], np.nan, np.nan]

        self.eclipse_params = {
            'primary_width': width1,
            'secondary_width': width2,
            'primary_position': pos1,
            'secondary_position': pos2,
            'primary_depth': depth1,
            'secondary_depth': depth2,
            'eclipse_edges': eclipse_edges
        }

        if interactive:
           self.interactive_eclipse()
        self.compute_eclipse_area(ecl=1)
        self.compute_eclipse_area(ecl=2)
        # self.eclipse_params = self.check_eclipses_credibility()

        # check if eclipses need to be swapped:
        if ~np.isnan(self.eclipse_params['primary_depth']) and ~np.isnan(self.eclipse_params['secondary_depth']):
            if self.eclipse_params['secondary_depth'] > self.eclipse_params['primary_depth']:
                pos1, d1, w1, edge1 = self.eclipse_params['secondary_position'], self.eclipse_params['secondary_depth'], self.eclipse_params['secondary_width'], self.eclipse_params['eclipse_edges'][2:]
                pos2, d2, w2, edge2 = self.eclipse_params['primary_position'], self.eclipse_params['primary_depth'], self.eclipse_params['primary_width'], self.eclipse_params['eclipse_edges'][:2]

                self.eclipse_params['primary_position'] = pos1 
                self.eclipse_params['primary_width'] = w1
                self.eclipse_params['primary_depth'] = d1
                self.eclipse_params['secondary_position'] = pos2 
                self.eclipse_params['secondary_width'] = w2
                self.eclipse_params['secondary_depth'] = d2
                self.eclipse_params['eclipse_edges'] = [edge1[0],edge1[1], edge2[0], edge2[1]]
        return self.eclipse_params

    @staticmethod
    def compute_gaussian_area(mu, sigma, d, xmin, xmax):
        '''
        Computes the area under an eclipse.

        The area is limited to an x-range of of [-2.8*sigma, 2.8*sigma] to correspond to
        the definition of an eclipse width of 5.6*.

        Parameters
        ----------
        mu: float
            Position (mean) of the Gaussian
        sigma: float
            Standard deviation of the Gaussian
        d: float
            Depth (amplitude) of the Gaussian
        xmin: float
            Bottom integral boundary
        xmax: float
            Top integral boundary
        
        Returns
        -------
        area: float
            The computed area under the Gaussian.

        '''
        if np.isnan(mu) or np.isnan(sigma) or np.isnan(d):
            return np.nan
        else:
            from scipy import special

            integral_top = special.erf((xmax - mu)/(np.sqrt(2)*sigma))
            integral_bottom = special.erf((xmin - mu)/(np.sqrt(2)*sigma))
            return np.sqrt(np.pi/2)*d*sigma*(integral_top-integral_bottom)


    def compute_eclipse_area(self, ecl=1):
        '''
        Computes the area under an eclipse.

        An eclipse is defined as being positioned at the Gaussian postion (mu),
        with a depth corresponding to the Gaussian amplitude (d) and width of 5.6*sigma.

        Parameters
        ----------
        ecl: int
            The eclipse whose area is to be computed (1 or 2)

        '''

        if hasattr(self, 'eclipse_area'):
            pass
        else:
            self.eclipse_area = {}

        if ecl==1:
            if self.best_fit['func'] in ['C', 'CE']:
                self.eclipse_area[ecl] = np.nan
            else:
                mu_ind = self.best_fit['param_names'].index('mu%i' % ecl)
                sigma_ind = self.best_fit['param_names'].index('sigma%i' % ecl)
                d_ind = self.best_fit['param_names'].index('d%i' % ecl)
                
                mu = self.best_fit['param_vals'][mu_ind]
                sigma = self.best_fit['param_vals'][sigma_ind]
                d = self.best_fit['param_vals'][d_ind]
                phi_top = mu + 2.8*sigma
                phi_bottom = mu - 2.8*sigma

                self.eclipse_area[ecl] = TwoGaussianModel.compute_gaussian_area(mu, sigma, d, phi_top, phi_bottom)
            
        else:
            if self.best_fit['func'] in ['C', 'CE', 'CG', 'CGE']:
                self.eclipse_area[ecl] = np.nan
            else:
                mu_ind = self.best_fit['param_names'].index('mu%i' % ecl)
                sigma_ind = self.best_fit['param_names'].index('sigma%i' % ecl)
                d_ind = self.best_fit['param_names'].index('d%i' % ecl)
                
                mu = self.best_fit['param_vals'][mu_ind]
                sigma = self.best_fit['param_vals'][sigma_ind]
                d = self.best_fit['param_vals'][d_ind]
                phi_top = mu + 2.8*sigma
                phi_bottom = mu - 2.8*sigma

                self.eclipse_area[ecl] = TwoGaussianModel.compute_gaussian_area(mu, sigma, d, phi_top, phi_bottom)


    def plot(self):
        import matplotlib.pyplot as plt

        plt.plot(self.phases, self.fluxes, 'b.')
        plt.plot(self.phases, self.model, 'r-')

        for i in range(4):
            plt.axvline(self.eclipse_params['eclipse_edges'][i])

        plt.show()


