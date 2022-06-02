import numpy as np
try:
    import emcee 
except:
    raise ImportError('emcee needs to be installed to run samplers.')
from phoebe.dependencies.ligeor.utils.lcutils import *
from multiprocessing import Pool

class EmceeSampler(object):

    def __init__(self, filename, times, fluxes, sigmas,
                period_init, t0_init, n_downsample, initial_fit_kwargs = {}, **kwargs):
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
        print(times)
        if len(times) == 0:
            if len(filename) == 0:
                raise ValueError("Must provide either a filename or data through times, fluxes and sigmas")
            else:
                self.__filename = filename
                self._period_init = period_init 
                self._t0_init = t0_init
                lc = load_lc(self.__filename, n_downsample=n_downsample, **kwargs)
                self._times, self._fluxes, self._sigmas = lc['times'], lc['fluxes'], lc['sigmas']
        else:
            # check length of phases and fluxes
            if len(times) != len(fluxes):
                raise ValueError('Length of array mismatch between times ({}) and fluxes ({})'.format(len(times), len(fluxes)))
            else:
                # sort just in case
                self._times, self._fluxes, self._sigmas = times[::n_downsample], fluxes[::n_downsample], sigmas[::n_downsample]

        self.initial_fit(**initial_fit_kwargs)
        
    def initial_fit(self, **kwargs):
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
        # overriden by subclass
        return None
    
    def logprob(self, values):
        '''
        Computes the logprobability of the sample.

        Parameters
        ----------
        values: array-like
            period (for phase folding) + model values
            
        '''
        # overriden by subclass
        return None

    def run_sampler(self, nwalkers=32, niters=2000, progress=True, emcee_kwargs = {}, sample_kwargs={}):
        '''
        Initializes and runs an emcee sampler.
        
        Parameters
        ----------
        nwalkers: int
            Number of walkers for emcee.
        niters: int
            Number of iterations to run.
        progress: bool
            If True, will output the progress (requires tqdm).
            
        Attributes
        ----------
        _sampler: object
            emcee.EnsembleSampler object, containing the MCMC samples, blobs and logprobabilities
        '''
        init_vals = self._initial_fit
        bestfit_vals = np.array([self._period_init, *init_vals])
        pos = bestfit_vals + 1e-4 * np.random.randn(nwalkers, len(bestfit_vals))
        nwalkers, ndim = pos.shape
        
        with Pool(1) as pool:
            self._sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logprob, pool=pool, **emcee_kwargs)
            self._sampler.run_mcmc(pos, niters, progress=progress, **sample_kwargs)
    

    def compute_results(self, burnin = 1000, save_lc=True, save_file='', show=False, failed=False):
        '''
        Computes a summary of the results from the sampler.

        The results computed include: ephemerides, model parameters and eclipse parameters. 

        Parameters
        ----------
        sampler: object
            The emcee sampler, initialized and run with .run_sampler()
        burnin: int
            Number of initial iterations to discard.
        save_lc: bool
            If True, will save the light curve to a file.
        save_file: str
            Filename to save to, if save_lc = True.
        show: bool
            If True, will show plot of the resulting light curve.
        failed: bool
            If True, all computed values are np.nan
            
        Attributes
        ----------
        _period_mcmc: dict
            Period mean and sigmas, computed from the posterior distribution
        _t0_mcmc: dict
            t0 mean and sigmas, computed from the posterior distribution
        _eclipse_params: dict
            Means of the eclipse parameters, computed from the posterior distribution
        _eclipse_params_errs: dict
            Uncertainities of the eclipse parameters from the posterior distribution
        _model_values: dict
            Model parameter values from the posterior distribution
        _model_values_errs: dict
            Uncertainities of the model parameter values from the posterior distribution
        _chi2: float
            Mean chi2 of the model, given the data.
        '''

        if save_lc and len(save_file) == 0:
            raise ValueError('Please provide a file name to save the model to or set save_lc=False.')
        
        # store the rest of the model parameters and their uncertainties
        means, sigmas_low, sigmas_high, means_blobs, sigmas_low_blobs, sigmas_high_blobs = self.compute_ephemerides(self._sampler, burnin, failed=failed)
        self.compute_model(means, sigmas_low, sigmas_high, save_lc = save_lc, save_file=save_file, show=show, failed=failed)
        self.compute_eclipse_params(means_blobs, sigmas_low_blobs, sigmas_high_blobs, failed=failed)


    def compute_ephemerides(self, sampler, burnin, failed=False):
        '''
        Computes mean and standard deviation for the period and t0 from the sample.

        Parameters
        ----------
        sampler: object
            The emcee sampler, initialized and run with .run_sampler()
        burnin: int
            Number of initial iterations to discard.
        '''

        if failed:
            self._period_mcmc = {'value': np.nan,
                    'sigma_low': np.nan, 
                    'sigma_high': np.nan}

            self._t0_mcmc = {'value': np.nan,
                    'sigma_low': np.nan, 
                    'sigma_high': np.nan}

            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        else:
            # process and log solution
            log_prob = sampler.get_log_prob(flat=True, discard=burnin)
            flat_samples = sampler.get_chain(flat=True, discard=burnin)
            flat_blobs = sampler.get_blobs(flat=True, discard=burnin)
            
            #figure out if there is branching in the solution and find the highest logp branch
            try:
                hist = np.histogram(log_prob, bins=50)
                arg_logp_max = np.argwhere(hist[0] != 0)[-1]
                logp_lim = hist[1][arg_logp_max-1]
                samples_top = flat_samples[log_prob >= logp_lim]
                blobs_top = flat_blobs[log_prob >= logp_lim]
                # log_prob_mean = np.mean(log_prob[log_prob >= logp_lim])
            except:
                samples_top = flat_samples
                blobs_top = flat_blobs
                # log_prob_mean = np.mean(log_prob)
            
            ndim = samples_top.shape[1]
            solution = []

            for j in range(ndim):
                mcmc = np.nanpercentile(samples_top[:, j], [16, 50, 84])
                q = np.diff(mcmc)
                solution.append([mcmc[1], q[0], q[1]])

            solution = np.array(solution)

            # compute the new period and errs
            means = solution[:,0]
            sigmas_low = solution[:,1]
            sigmas_high = solution[:,2]

            period = means[0]
            period_sigma_low = sigmas_low[0]
            period_sigma_high = sigmas_high[0]
            
            self._period_mcmc = {'value': period,
                                'sigma_low': period_sigma_low, 
                                'sigma_high': period_sigma_high}

            # compute the blob parameters
            # phasemin, residuals_mean, residuals_stdev, ecl1_area, ecl2_area
            ndim_blobs = blobs_top.shape[1]

            solution_blobs = []
            for j in range(ndim_blobs):
                mcmc_blob = np.nanpercentile(blobs_top[:, j], [16, 50, 84])
                q_blob = np.diff(mcmc_blob)
                solution_blobs.append([mcmc_blob[1], q_blob[0], q_blob[1]])

            solution_blobs = np.array(solution_blobs)
            means_blobs = solution_blobs[:,0]
            sigmas_low_blobs = solution_blobs[:,1]
            sigmas_high_blobs = solution_blobs[:,2]

            phasemin_mean = means_blobs[0]
            phasemin_sigma_low = sigmas_low_blobs[0]
            phasemin_sigma_high = sigmas_high_blobs[0]


            if np.isnan(self._t0_init):
                t0_new = 0 + period*phasemin_mean + int((self.__times.min()/period)+1)*(period)
            else:
                t0_new = self._t0_init + period*phasemin_mean

            t0_sigma_low = (period**2 * phasemin_sigma_low**2 + period_sigma_low**2 * phasemin_mean**2)**0.5
            t0_sigma_high = (period**2 * phasemin_sigma_high**2 + period_sigma_high**2 * phasemin_mean**2)**0.5

            self._t0_mcmc = {'value': t0_new,
                            'sigma_low': t0_sigma_low, 
                            'sigma_high': t0_sigma_high}

            return means, sigmas_low, sigmas_high, means_blobs, sigmas_low_blobs, sigmas_high_blobs


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
        # overriden by subclass
        return None


    def compute_eclipse_params(self, means, sigmas_low, sigmas_high, failed=False):
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
        failed: bool
            If true, all eclipse parameters are np.nan
        '''
        # pos1, width1, depth1, pos2, width2, depth2, ecl1_area, ecl2_area, residuals_mean, residuals_stdev
        eclipse_params = {'pos1': np.nan, 'width1': np.nan, 'depth1': np.nan, 
                          'pos2': np.nan, 'width2': np.nan, 'depth2': np.nan,
                          'ecl1_area': np.nan, 'ecl2_area': np.nan, 
                          'residuals_mean': np.nan, 'residuals_stdev': np.nan
                        }

        eclipse_params_err = {'pos1': np.nan, 'width1': np.nan, 'depth1': np.nan, 
                          'pos2': np.nan, 'width2': np.nan, 'depth2': np.nan,
                          'ecl1_area': np.nan, 'ecl2_area': np.nan, 
                          'residuals_mean': np.nan, 'residuals_stdev': np.nan
                        }

        if not failed:
            for ind, eclkey in enumerate(eclipse_params.keys()):
                # print(eclkey, means[ind], np.max((sigmas_low[ind],sigmas_high[ind])))
                eclipse_params[eclkey] = means[ind]
                eclipse_params_err[eclkey] = np.max((sigmas_low[ind],sigmas_high[ind]))

        
        self._eclipse_params = eclipse_params
        self._eclipse_params_errs = eclipse_params_err
        

    def save_results_to_file(self, results_file, type='ephemerides', ind=''):
        '''
        Save the resulting ephemerides, model or eclipse parameters to a file.

        Parameters
        ----------
        results_file: str
            Filename to save to
        type: str
            Which parameters to store. Available choices: ['ephemerides', 'model_values', 'eclipse_parameters']
        ind: str
            Index of the object (if looping through a list, otherwise optional)
        '''

        if type=='ephemerides':
            with open(results_file, 'a') as f:
                f.write('{},{},{},{},{},{},{},{}\n'.format(ind,
                                                            self._period_mcmc['value'],
                                                            self._period_mcmc['sigma_low'],
                                                            self._period_mcmc['sigma_high'],
                                                            self._t0_mcmc['value'],
                                                            self._t0_mcmc['sigma_low'],
                                                            self._t0_mcmc['sigma_high'],
                                                            self._chi2))

        elif type=='model_values':
            results_str = self._func+','

            for i,mkey in enumerate(self._model_values.keys()):
                results_str += '{},{}'.format(self._model_values[mkey],self._model_values_errs[mkey])
                if i < len(self._model_values.keys())-1:
                    results_str += ','
                else:
                    results_str += '\n'

            with open(results_file, 'a') as f:
                f.write('{},{}'.format(ind,results_str))


        elif type == 'eclipse_parameters':
            with open(results_file, 'a') as f:
                # pos1, pos1_err, width1, width1_err, depth1, depth1_err, 
                # pos2, pos2_err, width2, width2_err, depth2, depth2_err, 
                # res_mean, res_mean_err, res_stdev, res_stdev_err, 
                # ecl1_area, ecl1_area_err, ecl2_area, ecl2_area_err
                ecl = self._eclipse_params
                eclerr = self._eclipse_params_errs
                f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(ind,
                    ecl['pos1'], eclerr['pos1'], ecl['width1'], eclerr['width1'], ecl['depth1'], eclerr['depth1'],
                    ecl['pos2'], eclerr['pos2'], ecl['width2'], eclerr['width2'], ecl['depth2'], eclerr['depth2'],
                    ecl['residuals_mean'], eclerr['residuals_mean'], ecl['residuals_stdev'], eclerr['residuals_stdev'],
                    ecl['ecl1_area'], eclerr['ecl1_area'], ecl['ecl2_area'], eclerr['ecl2_area']
                ))

        else:
            raise NotImplementedError