import numpy as np
from phoebe.dependencies.ligeor.fitting.sampler import EmceeSampler
from phoebe.dependencies.ligeor.models.twogaussian import TwoGaussianModel
from phoebe.dependencies.ligeor.utils.lcutils import *


class EmceeSamplerTwoGaussian(EmceeSampler):
    
    def __init__(self, filename='', times = [], fluxes = [], sigmas = [],
                period_init=1, t0_init=0, n_downsample=0, **kwargs):
        '''
        Initializes a sampler for the light curve stored in 'filename'
        with determined initial values for the period and t0.

        Parameters
        ----------
        filename: str
            Filename to load the raw light curve from (expected format: times, fluxes, sigmas)
        times: array-like
            Time points of the observations
        fluxes: array-like
            Observed fluxes
        sigmas: array-like
            Uncertainities of the observed fluxes
        period_init: float
            Initial value of the period (from code/triage)
        t0_init: float
            Initial value of the time of superior conjunction (t0)
        n_downsample: int
            Number of data points to skip in raw light curve for downsampling
        nbins: int
            Number of bins (applies to the computed final model).
        
        Keyword arguments are passed on to load_lc.
            
        
        Attributes
        ----------
        _period_init: float
            Initial value of the orbital period
        _t0_init: float
            Initial value of the time of superior conjunction
        '''
        
        super(EmceeSamplerTwoGaussian,self).__init__(filename, times, fluxes, sigmas, period_init, t0_init, n_downsample, **kwargs)


    def initial_fit(self):
        '''
        Runs an initial fit to the data with the chosen model (two-Gaussian or polyfit).
        
        Attributes
        ----------
        _func: str
            Name of the best fit model function.
        _model_params: array-like
            Labels of all model parameters
        _initial_fit: array-like
            Initial values of the model parameters from optimization.
        '''
        phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                        self._fluxes, 
                                                        self._sigmas, 
                                                        period=self._period_init, 
                                                        t0=self._t0_init,
                                                        interval = '05')

        twogModel = TwoGaussianModel(phases=phases, 
                                    fluxes=fluxes_ph, 
                                    sigmas=sigmas_ph, 
                                    )
        twogModel.fit()

        # self._twogModel_init = twogModel
        self._func = twogModel.best_fit['func']
        self._model_params = twogModel.best_fit['param_names']
        self._initial_fit = twogModel.best_fit['param_vals']


    def logprob(self, values):
        '''
        Computes the logprobability of the sample.

        Parameters
        ----------
        values: array-like
            period (for phase folding) + model values
        '''

        fmax = self._fluxes.max()
        fmin = self._fluxes.min()
        fdiff = fmax - fmin

        bounds = {'C': ((0),(fmax)),
            'CE': ((0, 1e-6, -0.5),(fmax, fdiff, 0.5)),
            'CG': ((0., -0.5, 0., 0.), (fmax, 0.5, fdiff, 0.5)),
            'CGE': ((0., -0.5, 0., 0., 1e-6, -0.5),(fmax, 0.5, fdiff, 0.5, fdiff, 0.5)),
            'CG12': ((0.,-0.5, 0., 0., -0.5, 0., 0.),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5)),
            'CG12E1': ((0.,-0.5, 0., 0., -0.5, 0., 0., 1e-6),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5, fdiff)),
            'CG12E2': ((0.,-0.5, 0., 0., -0.5, 0., 0., 1e-6),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5, fdiff))}
        
        period, *model_vals = values

        for i,param_val in enumerate(model_vals):
            if param_val < bounds[self._func][0][i] or param_val > bounds[self._func][1][i]:
                # raise Warning('out of prior', self._func, bounds[self._func][0][i], bounds[self._func][1][i], param_val)
                return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
        # fold with period
        phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                        self._fluxes, 
                                                        self._sigmas, 
                                                        period=period, 
                                                        t0=self._t0_init,
                                                        interval='05')
        # compute model with the selected twog function
        # TODO: review this part for potentially more efficient option
        twog = TwoGaussianModel(phases=phases, 
                                    fluxes=fluxes_ph, 
                                    sigmas=sigmas_ph)
        twog.fit(fit_funcs=[self._func], param_vals=[model_vals])
        
        eclipse_dir = twog.compute_eclipse_params()
        pos1, pos2 = eclipse_dir['primary_position'], eclipse_dir['secondary_position']
        width1, width2 = eclipse_dir['primary_width'], eclipse_dir['secondary_width']
        depth1, depth2 = eclipse_dir['primary_depth'], eclipse_dir['secondary_depth']
        ecl1_area, ecl2_area = twog.eclipse_area[1], twog.eclipse_area[2]
        residuals_mean, residuals_stdev = compute_residuals_stdev(fluxes_ph, twog.model)

        logprob = -0.5*(np.sum((fluxes_ph-twog.model)**2/sigmas_ph**2))
        # print(logprob, pos1, width1, depth1, pos2, width2, depth2)#, ecl1_area, ecl2_area, residuals_mean, residuals_stdev)
        return logprob, pos1, width1, depth1, pos2, width2, depth2, ecl1_area, ecl2_area, residuals_mean, residuals_stdev

    
    def compute_model(self, means, sigmas_low, sigmas_high, save_lc = True, save_file='', show=False, failed=False):
        '''
        Computes the model parameter values from the sample.

        Parameters
        ----------
        means: array-like
            Mean values from the sample
        sigmas_low: array-like
            Standard deviation of samples < mean
        sigmas_high: array_like
            Standard deviation of samples > mean
        save_lc: bool
            If True, saves the model light curve to a file
        save_file: str
            File name to save light curve to, if save_lc=True.
        show: bool
            If True, will display a plot of the model light curve.
        failed: bool
            If True, all model parameters are np.nan
        '''        
        
        model_results = {'C': np.nan, 'mu1': np.nan, 'd1': np.nan, 'sigma1': np.nan, 
                        'mu2': np.nan, 'd2': np.nan, 'sigma2': np.nan, 'Aell': np.nan, 'phi0': np.nan
                        }
        model_results_err = {'C': np.nan, 'mu1': np.nan, 'd1': np.nan, 'sigma1': np.nan, 
                        'mu2': np.nan, 'd2': np.nan, 'sigma2': np.nan, 'Aell': np.nan, 'phi0': np.nan
                        }
        
        # results_str = '{}'.format(func)
        if failed:
            self._model_values = model_results
            self._model_values_errs = model_results_err
            self._chi2 = np.nan
        
        else:
            for mkey in model_results.keys():
                if mkey in self._model_params:
                    pind = self._model_params.index(mkey)
                    model_results[mkey] = means[pind+1]
                    model_results_err[mkey] = np.max((sigmas_low[pind+1],sigmas_high[pind+1]))
                    # results_str += ',{},{}'.format(model_results[mkey],model_results_err[mkey])
                
            self._model_values = model_results
            self._model_values_errs = model_results_err
            chi2 = np.nan

            phases_obs, fluxes_ph_obs, sigmas_ph_obs = phase_fold(self._times, 
                                                    self._fluxes, 
                                                    self._sigmas, 
                                                    period=self._period_mcmc['value'], 
                                                    t0=self._t0_init, interval='05')
            twog_func = getattr(TwoGaussianModel, self._func.lower())
            fluxes_model = twog_func(phases_obs, *means[1:])
            chi2 = -0.5*(np.sum((fluxes_ph_obs-fluxes_model)**2/sigmas_ph_obs**2))
            
            if show:
                import matplotlib.pyplot as plt
                plt.plot(phases_obs, fluxes_ph_obs, 'k.')
                plt.plot(phases_obs, fluxes_model, 'r-')
                plt.show()

            if save_lc:
                np.savetxt(save_file, np.array([phases_obs, fluxes_model]).T)

            self._chi2 = chi2