import numpy as np

def load_lc(lc_file, n_downsample=0, phase_folded=False, usecols=(0,1,2), delimiter=','):
    '''
    Loads the light curve from lc_file and returns 
    separate arrays for times, fluxes and sigmas.

    Parameters
    ----------
    lc_file: str
        Filename to load light curve from.
    n_downsample: int
        Number of data points to skip in downsampling the lc.

    Returns
    -------
    A dictionary of the times, fluxes and sigmas retrieved from the lc file.
    '''
    
    lc = np.loadtxt(lc_file, usecols=usecols, delimiter=delimiter)

    if phase_folded:
        return {'phases': lc[:,0][::n_downsample], 
        'fluxes': lc[:,1][::n_downsample], 
        'sigmas': lc[:,2][::n_downsample]}

    else:
        return {'times': lc[:,0][::n_downsample], 
                'fluxes': lc[:,1][::n_downsample], 
                'sigmas': lc[:,2][::n_downsample]}


def phase_fold(times, fluxes, sigmas, period=1, t0=0, interval='05'):
    '''
    Phase-folds the light curve with a given period and t0.

    Parameters
    ----------
    times: array-like
        The observation times
    fluxes: array-like
        Observed fluxes corresponding to each time in times
    sigmas: array-like
        Uncertainties corresponding to each flux in fluxes
    interval: str
        Phase range to be returned, [-0.5,0.5] if interval='05' or [0.,1.] if interval='01'.


    Returns
    -------
    phases: array-like
        The computed orbital phases on a range [-0.5,0.5], sorted in ascending order
    fluxes_ph: array_like
        The fluxes resorted to match the phases array order
    sigmas_ph: array_like
        The sigmas resorted to match the phases array order
    '''
    
    t0 = 0 if np.isnan(t0) else t0
    phases = np.mod((times-t0)/period, 1.0)

    if interval == '05':
        if isinstance(phases, float):
            if phases > 0.5:
                phases -= 1
        else:
            # then should be an array
            phases[phases > 0.5] -= 1
        
    s = phases.argsort()
    phases = phases[s]
    fluxes_ph = fluxes[s]
    sigmas_ph = sigmas[s]
    
    return phases, fluxes_ph, sigmas_ph


def extend_phasefolded_lc(phases, fluxes, sigmas):
    '''
    Takes a phase-folded light curve on the range [-0.5,0.5] and extends it on range [-1,1]
    
    Parameters
    ----------
    phases: array-like
        Array of input phases spanning the range [-0.5,0.5]
    fluxes: array-like
        Corresponding fluxes, length must be equal to that of phases
    sigmas: array-like
        Corresponsing measurement uncertainties, length must be equal to that og phases
        
    Returns
    -------
    phases_extend, fluxes_extend, sigmas_extend: array-like
        Extended arrays on phase-range [-1,1]
    
    '''
    #make new arrays that would span phase range -1 to 1:
    fluxes_extend = np.hstack((fluxes[(phases > 0)], fluxes, fluxes[phases < 0.]))
    phases_extend = np.hstack((phases[phases>0]-1, phases, phases[phases<0]+1))

    if sigmas is not None:
        sigmas_extend = np.hstack((sigmas[phases > 0], sigmas, sigmas[phases < 0.]))
    else:
        sigmas_extend = None

    return phases_extend, fluxes_extend, sigmas_extend


def check_eclipse_fitting_noise(fluxes_model, fluxes_obs, depth):
    '''
    Checks if the model eclipses are true or fit data noise features.

    Parameters
    ----------
    fluxes_model: array-like
        Fluxes computed from the model
    fluxes_obs: array-like
        Observed fluxes
    depth: float
        Depth of the eclipse

    Returns
    -------
    bool
    '''

    sigma_res = np.std(fluxes_obs - fluxes_model)
    if depth <= 0.5*sigma_res:
        return True
    else:
        return False


def check_eclipse_fitting_cosine(width):
    '''
    Checks if an eclipse is fitted to the out-of-eclipse variability.

    Parameters
    ----------
    width: float
        Measured width of the eclipse from the model.

    Returns
    -------
    bool
    '''
    if width >= 0.41:
        return True
    else:
        return False

def check_eclipse_depth_negative(depth):
    '''
    Checks if an eclipse is fitting another feature that does not show a decrease in flux.
    Parameters
    ----------
    depth: float
        Measured depth of the eclipse from the model.

    Returns
    -------
    bool
    '''
    if depth <= 0.0:
        return True
    else:
        return False


def compute_residuals_stdev(fluxes_obs, fluxes_model):
    '''
    Computes the residuals of the input fluxes and best fit model

    Parameters
    ----------
    fluxes_obs: array-like
        Observed fluxes
    fluxes_model: array-like
        Model fluxes

    Returns
    -------
    residuals_mean: float
        The mean of the residuals
    residuals_stdev: float
        The standard deviation of the residuals
    '''
    residuals = fluxes_obs - fluxes_model
    return np.mean(residuals), np.std(residuals)