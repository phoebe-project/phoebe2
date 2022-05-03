import numpy as np
from phoebe.dependencies import distl

def sample_skewed_gaussian(mean, sigma_low, sigma_high, size=1000):
    '''
    Samples from a skewed Gaussian distribution (hacky).
    
    Assumes different sigma left and right from the mean, samples them separately
    and combines into one sample, stitched together at the mean.
    
    Parameters
    ----------
    mean: float
        Mean of the Gaussian distribution
    sigma_low: float
        Standard deviation of the left Gaussian
    sigma_high: float
        Standard deviation of the right Gaussian
    size: int
        Number of samples to return
        
    Returns
    -------
    An array of the samples drawn from the distribution.
    '''
    
    samples_low = distl.gaussian(mean, sigma_low).sample(size)
    samples_high = distl.gaussian(mean, sigma_high).sample(size)
    
    samples_low_cut = samples_low[samples_low < mean]
    samples_high_cut = samples_high[samples_high >= mean]
    
    return np.hstack((samples_low_cut, samples_high_cut))

def combine_dists_from_hist(samples1, samples2, bins=1000, plot=False):
    '''
    Combine samples from two models and computes new mean and standard deviation.
    
    Parameters
    ----------
    samples1: array-like
        Samples drawn from first distribution.
    samples2: array-like
        Samples drawn from second distribution.
    bins: int
        Number of bins for the histogram
    plot: bool
        If True, will display the histogram.
        
    Returns
    -------
    A dictionary with the mean, sigma_low, sigma_high and confidence of the combined
    distribution.
    '''
    
    samples_full = np.hstack((samples1, samples2))
    hist_combined = distl.histogram_from_data(samples_full, bins=bins)
    
    if plot:
        hist_combined.plot()
        
    uncs = hist_combined.uncertainties()
    return {'mean': uncs[1], 
            'sigma_low': uncs[1]-uncs[0], 
            'sigma_high': uncs[2]-uncs[1],
            'confidence': 1}

def compute_combined_2g_pf_value(value_2g=1, value_2g_sigma_low=0, value_2g_sigma_high=0, weight_2g = 0.5,
                            value_pf=1, value_pf_sigma_low=0, value_pf_sigma_high=0, weight_pf = 0.5,
                            nsamples = 10000):
    '''
    Compute the combined value from two-Gaussian and polyfit MCMC results.
    
    Parameters
    ----------
    value_2g: float
        TwoGaussian value mean
    value_2g_sigma_low: float
        TwoGaussian value std for samples < value_2g
    value_2g_sigma_high: float
        TwoGaussian value std for samples > value_2g
    weight_2g: float
        The ratio of samples to be drawn from the TwoGaussian distribution, typically 1/chi2_2g
    value_pf: float
        Polyfit value mean
    value_pf_sigma_low: float
        Polyfit value std for samples < value_pf
    value_pf_sigma_high: float
        Polyfit value std for samples > value_pf
    weight_pf: float
        The ratio of samples to be drawn from the Polyfit distribution, typically 1/chi2_pf
    nsamples: int
        Number of samples for the combined distribution.   
        
    Returns
    -------
    value_mean: float
        The mean calculated from the combined distributions.
    value_sigma_low: float
        The standard deviation of the samples < mean
    value_sigma_high: float
        The standard deviation of the samples > mean 
    '''
    # from 2g and pf to combined values
        
    # print('2g fit: P = %.5f + %.5f - %.5f' % (value_2g, value_2g_sigma_high, value_2g_sigma_low))
    # print('pf fit: P = %.5f + %.5f - %.5f' % (value_pf, value_pf_sigma_high, value_pf_sigma_low))
    
    if ~np.isnan(weight_2g) and ~np.isnan(weight_pf):
        wratio = weight_2g/weight_pf

        nsamples_2g = int(wratio*nsamples/(1+wratio))
        nsamples_pf = int(nsamples/(1+wratio))

        # print('values check: w_2g = %.6f, w_pf=%.6f, N_2g = %i, N_pf = %i'
        #  % (weight_2g, weight_pf, nsamples_2g, nsamples_pf))

        samples_2g = sample_skewed_gaussian(value_2g, value_2g_sigma_low, value_2g_sigma_high, size=nsamples_2g)
        samples_pf = sample_skewed_gaussian(value_pf, value_pf_sigma_low, value_pf_sigma_high, size=nsamples_pf)

        combined_result = combine_dists_from_hist(samples_2g, samples_pf, bins=1000, plot=False)
        return combined_result['mean'], combined_result['sigma_low'], combined_result['sigma_high']
        

    elif np.isnan(weight_2g) and ~np.isnan(weight_pf):
        return value_pf, value_pf_sigma_low, value_pf_sigma_high

        
    elif ~np.isnan(weight_2g) and np.isnan(weight_pf):
        return value_2g, value_2g_sigma_low, value_2g_sigma_high

    
    else:
        return np.nan, np.nan, np.nan