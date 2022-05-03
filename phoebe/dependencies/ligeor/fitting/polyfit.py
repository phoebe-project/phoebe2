import numpy as np
from phoebe.dependencies.ligeor.fitting.sampler import EmceeSampler
from phoebe.dependencies.ligeor.models.polyfit import Polyfit
from phoebe.dependencies.ligeor.utils.lcutils import *


class EmceeSamplerPolyfit(EmceeSampler):
    
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
        
        super(EmceeSamplerPolyfit,self).__init__(filename, times, fluxes, sigmas, period_init, t0_init, n_downsample, **kwargs)


    def initial_fit(self, knots = [], coeffs = []):
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
        
        if len(knots) == 0:
            phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                            self._fluxes, 
                                                            self._sigmas, 
                                                            period=self._period_init, 
                                                            t0=self._t0_init, interval='01')

            polyfit = Polyfit(phases=phases, 
                                        fluxes=fluxes_ph, 
                                        sigmas=sigmas_ph, 
                                        )
            polyfit.fit()

            self._initial_fit = np.hstack((np.array(polyfit.knots), np.array(polyfit.coeffs).reshape(12)))

        else:
            if len(coeffs) == 0:
                phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                            self._fluxes, 
                                                            self._sigmas, 
                                                            period=self._period_init, 
                                                            t0=self._t0_init, interval='01')

                polyfit = Polyfit(phases=phases, 
                                            fluxes=fluxes_ph, 
                                            sigmas=sigmas_ph, 
                                            )
                polyfit.fit(knots = knots)
                self._initial_fit = np.hstack((np.array(knots), np.array(polyfit.coeffs).reshape(12)))
            
            else:
                self._initial_fit = np.hstack((np.array(knots), np.array(coeffs).reshape(12)))

        self._func = 'PF'
        self._model_params = ['k1', 'k2', 'k3', 'k4', 'c11', 'c12', 'c13', 'c21', 'c22', 'c23', 'c31', 'c32', 'c33', 'c41', 'c42', 'c43']
        self._bounds = []
        for i,value in enumerate(self._initial_fit):
            self._bounds.append([value-0.1,value+0.1*value])


    def logprob(self, values):
        '''
        Computes the logprobability of the sample.

        Parameters
        ----------
        values: array-like
            period (for phase folding) + model values
        '''
        # bounds = [[-1e-5,-1e-5,-1e-5,-1e-5],[1+1e-5,1+1e-5,1+1e-5,1+1e-5]]
        # bounds = [[-1,-1,-1,-1],[2,2,2,2]]
        period, *model_vals = values

        for i,param_val in enumerate(model_vals[:4]):
            if param_val < self._bounds[i][0] or param_val > self._bounds[i][1]:
                # print('out of prior', self._bounds[i][0], self._bounds[i][1], param_val)
                return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                        self._fluxes, 
                                                        self._sigmas, 
                                                        period=period, 
                                                        t0=self._t0_init, interval='01')
        
        try:
            #TODO: figure out the cause for the "cannot unpack error"
            polyfit = Polyfit(phases=phases, 
                                        fluxes=fluxes_ph, 
                                        sigmas=sigmas_ph)
            polyfit.fit(knots = np.array(model_vals[:4]), coeffs = np.array(model_vals[4:]).reshape(4,3))
            
            eclipse_dir = polyfit.compute_eclipse_params()
            pos1, pos2 = eclipse_dir['primary_position'], eclipse_dir['secondary_position']
            width1, width2 = eclipse_dir['primary_width'], eclipse_dir['secondary_width']
            depth1, depth2 = eclipse_dir['primary_depth'], eclipse_dir['secondary_depth']
            ecl1_area, ecl2_area = polyfit.eclipse_area[1], polyfit.eclipse_area[2]
            residuals_mean, residuals_stdev = compute_residuals_stdev(fluxes_ph, polyfit.model)

            # print('residuals: ', residuals_mean, residuals_stdev)
            logprob = -0.5*(np.sum((fluxes_ph-polyfit.model)**2/sigmas_ph**2))
            # print(logprob, pos1, width1, depth1, pos2, width2, depth2)#, ecl1_area, ecl2_area, residuals_mean, residuals_stdev)
            return logprob, pos1, width1, depth1, pos2, width2, depth2, ecl1_area, ecl2_area, residuals_mean, residuals_stdev
        except:
            return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


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
                
        model_results = {'k1': np.nan, 'k2': np.nan, 'k3': np.nan, 'k4': np.nan,
                        'c11': np.nan, 'c12': np.nan, 'c13': np.nan,
                        'c21': np.nan, 'c22': np.nan, 'c23': np.nan,
                        'c31': np.nan, 'c32': np.nan, 'c33': np.nan,
                        'c41': np.nan, 'c42': np.nan, 'c43': np.nan
                        }
        model_results_err = {'k1': np.nan, 'k2': np.nan, 'k3': np.nan, 'k4': np.nan,
                            'c11': np.nan, 'c12': np.nan, 'c13': np.nan,
                            'c21': np.nan, 'c22': np.nan, 'c23': np.nan,
                            'c31': np.nan, 'c32': np.nan, 'c33': np.nan,
                            'c41': np.nan, 'c42': np.nan, 'c43': np.nan
                        }
        if failed:
            self._model_values = model_results
            self._model_values_errs = model_results_err
            self._chi2 = np.nan

        else:
            # results_str = '{}'.format(func)
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
                                                    t0=self._t0_init,
                                                    interval='01')
        
            polyfit = Polyfit(phases=phases_obs, 
                                    fluxes=fluxes_ph_obs, 
                                    sigmas=sigmas_ph_obs)

            knots = np.array(list(self._model_values.values())[:4])
            coeffs = np.array(list(self._model_values.values())[4:]).reshape(4,3)
            polyfit.fit(knots = knots, coeffs = coeffs)
            fluxes_model = polyfit.fv(x = phases_obs)
            chi2 = -0.5*(np.sum((fluxes_ph_obs-fluxes_model)**2/sigmas_ph_obs**2))
            self._chi2 = chi2

            if show:
                import matplotlib.pyplot as plt
                plt.plot(phases_obs, fluxes_ph_obs, 'k.')
                plt.plot(phases_obs, fluxes_model, 'r-')
                plt.show()

            if save_lc:
                np.savetxt(save_file, np.array([phases_obs, fluxes_model]).T)