import numpy as np
from numpy import sin, cos, pi, sqrt, tan, arccos, arcsin
try:
    from scipy.optimize import newton, minimize, curve_fit
    from scipy.signal import find_peaks, savgol_filter
except ImportError:
    _can_compute_eclipse_params = False
else:
    _can_compute_eclipse_params = True

import logging
logger = logging.getLogger("SOLVER")
logger.addHandler(logging.NullHandler())

# two-Gaussian model stuff

# HELPER FUNCTIONS

def ellipsoidal(phi, Aell, phi0):
    return 0.5*Aell*np.cos(4*np.pi*(phi-phi0))

def gaussian(phi, mu, d, sigma):
    return d*np.exp(-(phi-mu)**2/(2*sigma**2))

def gsum(phi, mu, d, sigma):
    gauss_sum = np.zeros(len(phi))
    for i in range(-2,3,1):
        gauss_sum += gaussian(phi,mu+i,d,sigma)
    return gauss_sum

# CHOICE OF MODELS

def const(phi, C):
    return C*np.ones(len(phi))
    
def ce(phi, C, Aell, phi0):
    return const(phi, C) - ellipsoidal(phi, Aell, phi0)

def cg(phi, C, mu, d,  sigma):
    return const(phi, C) - gsum(phi, mu, d, sigma)
    
def cge(phi, C, mu, d, sigma, Aell):
    return const(phi, C) - ellipsoidal(phi, Aell, mu) - gsum(phi, mu, d, sigma)

def cg12(phi, C, mu1, d1, sigma1, mu2, d2, sigma2):
    return const(phi, C) - gsum(phi, mu1, d1, sigma1) - gsum(phi, mu2, d2, sigma2)

def cg12e1(phi, C, mu1, d1, sigma1, mu2, d2, sigma2, Aell):
    return const(phi, C) - gsum(phi, mu1, d1, sigma1) - gsum(phi, mu2, d2, sigma2) - ellipsoidal(phi, Aell, mu1)

def cg12e2(phi, C, mu1, d1, sigma1, mu2, d2, sigma2, Aell):
    return const(phi, C) - gsum(phi, mu1, d1, sigma1) - gsum(phi, mu2, d2, sigma2) - ellipsoidal(phi, Aell, mu2)


# PREPROCESSING

def extend_phasefolded_lc(phases, fluxes, sigmas):
    
    #make new arrays that would span phase range -1 to 1:
    fluxes_extend = np.hstack((fluxes[(phases > 0)], fluxes, fluxes[phases < 0.]))
    sigmas_extend = np.hstack((sigmas[phases > 0], sigmas, sigmas[phases < 0.]))
    phases_extend = np.hstack((phases[phases>0]-1, phases, phases[phases<0]+1))
    
    return phases_extend, fluxes_extend, sigmas_extend

def estimate_eclipse_phases(phases, fluxes, smooth = False):

    if smooth:
        wl = int(len(fluxes)/50)
        if wl%2 == 0:
            wl = wl+1
        po = 3 if wl > 3 else 1
        lc_ph = savgol_filter(fluxes, window_length=wl, polyorder=po)
    else:
        lc_ph = fluxes
    
    ph = phases
    
    lc_peaks = lc_ph[np.argwhere(lc_ph < lc_ph.mean())].flatten()
    ph_peaks = ph[np.argwhere(lc_ph < lc_ph.mean())].flatten()
    peaks = find_peaks(-lc_peaks)

    filter_positive_phases = peaks[0][(ph_peaks[peaks[0]]>=-0.05) & (ph_peaks[peaks[0]]<0.95)]
    phases_ecl = ph_peaks[filter_positive_phases][np.argsort(lc_peaks[filter_positive_phases])][:2]
    phases_ecl = phases_ecl[np.argsort(phases_ecl)]

    if len(phases_ecl) == 2:
        mu1 = phases_ecl[0]
        mu2 = phases_ecl[1]
    elif len(phases_ecl) == 1:
        mu1 = phases_ecl[0]
        mu2 = 0.5
    else:
        mu1 = 0.
        mu2 = 0.5
    return mu1, mu2

# FITTING

def lnlike(y, yerr, ymodel):
    return -np.sum(np.log((2*np.pi)**0.5*yerr)+(y-ymodel)**2/(2*yerr**2))


def bic(y, yerr, ymodel, nparams):
    return 2*lnlike(y,yerr,ymodel) - nparams*np.log(len(y))


def fit_twoGaussian_models(phases, fluxes, sigmas, smooth=False):
    # extend light curve on phase range [-1,1]
    phases, fluxes, sigmas = extend_phasefolded_lc(phases, fluxes, sigmas)
    # setup the initial parameters
    
    # fit all of the models to the data
    twogfuncs = {'C': const, 'CE': ce, 'CG': cg, 'CGE': cge, 'CG12': cg12, 'CG12E1': cg12e1, 'CG12E2': cg12e2}

    C0 = fluxes.max()
    mu10, mu20 = estimate_eclipse_phases(phases, fluxes, smooth=smooth)
    d10 = fluxes.max()-fluxes[np.argmin(np.abs(phases-mu10))]
    d20 = fluxes.max()-fluxes[np.argmin(np.abs(phases-mu20))]
    sigma10, sigma20 = 0.05, 0.05
    Aell0 = 0.001
    init_params = {'C': [C0,], 
        'CE': [C0, Aell0, mu10], 
        'CG': [C0, mu10, d10, sigma10], 
        'CGE': [C0, mu10, d10, sigma10, Aell0], 
        'CG12': [C0, mu10, d10, sigma10, mu20, d20, sigma20], 
        'CG12E1': [C0, mu10, d10, sigma10, mu20, d20, sigma20, Aell0], 
        'CG12E2': [C0, mu10, d10, sigma10, mu20, d20, sigma20, Aell0]}
    
    fits = {}
    for key in twogfuncs.keys():
        try:
            fits[key] = curve_fit(twogfuncs[key], phases, fluxes, p0=init_params[key], sigma=sigmas)
        except:
            fits[key] = np.array([np.nan*np.ones(len(init_params[key]))])

    return fits


def compute_twoGaussian_models(fits, phases):
    twogfuncs = {'C': const, 'CE': ce, 'CG': cg, 'CGE': cge, 'CG12': cg12, 'CG12E1': cg12e1, 'CG12E2': cg12e2}
        
    models = {}

    for fkey in fits.keys():
        models[fkey] = twogfuncs[fkey](phases, *fits[fkey][0])
        
    return models


def compute_twoGaussian_models_BIC(models, phases, fluxes, sigmas):
    bics = {}
    nparams = {'C':1, 'CE':3, 'CG':4, 'CGE':5, 'CG12':7, 'CG12E1':8, 'CG12E2':8}
    
    for mkey in models.keys():
        bics[mkey] = bic(fluxes, sigmas, models[mkey], nparams[mkey])
        
    return bics


def fit_lc(phases, fluxes, sigmas, smooth=False):

    fits = fit_twoGaussian_models(phases, fluxes, sigmas, smooth=smooth)
    models = compute_twoGaussian_models(fits, phases)
    bics = compute_twoGaussian_models_BIC(models, phases, fluxes, sigmas)
    params = {'C': ['C'], 
            'CE': ['C', 'Aell', 'mu1'], 
            'CG': ['C', 'mu1', 'd1', 'sigma1'], 
            'CGE': ['C', 'mu1', 'd1', 'sigma1', 'Aell'], 
            'CG12': ['C', 'mu1', 'd1', 'sigma1', 'mu2', 'd2', 'sigma2'], 
            'CG12E1': ['C', 'mu1', 'd1', 'sigma1', 'mu2', 'd2', 'sigma2', 'Aell'], 
            'CG12E2': ['C', 'mu1', 'd1', 'sigma1', 'mu2', 'd2', 'sigma2', 'Aell']}
    
    best_fit = list(models.keys())[np.argmax(list(bics.values()))]
    return {'fits':fits, 'models':models, 'bics':bics, 'best_fit':best_fit, 'model_parameters': params}

# REFINING THE FIT

def two_line_model(values,breakpoint,x,y,sigma):
    k1, n1, k2, n2 = values
    ymodel = np.zeros(len(x))
    ymodel[x<=breakpoint] = k1*x[x<=breakpoint] + n1
    ymodel[x>breakpoint] = k2*x[x>breakpoint]+n2
    return np.sum((ymodel-y)**2/sigma**2)


def refine_eclipse_widths(phases, fluxes, sigmas, pos1, pos2, width1, width2, wf1=0.75, wf2=0.25):

    # mask out regions around the eclipses
    mask11 = (pos1 - wf1*width1 < phases) & (phases < pos1-wf2*width1) 
    mask12 = (phases < pos1 + wf1*width1) & (phases > pos1+wf2*width1) 
    mask21 = (pos2 - wf1*width2 < phases) & (phases < pos2-wf2*width2) 
    mask22 = (phases < pos2 + wf1*width2) & (phases > pos2+wf2*width1) 

    eclipse_breaks = np.zeros(4)

    try:
        for i,mask in enumerate([mask11, mask12, mask21, mask22]):
            breakpoints = np.linspace(phases[mask].min(), phases[mask].max(), len(phases[mask]))
            chis2 = np.zeros(len(breakpoints))
            for j,breakp in enumerate(breakpoints):
                sol = minimize(two_line_model, [0., 2., 0., 2.], args=(breakp, phases[mask], fluxes[mask], sigmas[mask]))
                k1, n1, k2, n2 = sol['x']
                ymodel = np.zeros(len(phases[mask]))
                ymodel[phases[mask]<=breakp] = k1*phases[mask][phases[mask]<=breakp] + n1
                ymodel[phases[mask]>breakp] = k2*phases[mask][phases[mask]>breakp]+n2
                chis2[j] = np.sum((ymodel-fluxes[mask])**2)
            eclipse_breaks[i] = phases[mask][np.argmin(chis2)]

        return eclipse_breaks
    except:
        logger.warning('Eclipse width refinement failed.')
        return None

# GEOMETRY SOLVER

def compute_eclipse_params(phases, fluxes, sigmas, smooth=False, diagnose=False):

    fit_result = fit_lc(phases, fluxes, sigmas, smooth=smooth)
    best_fit = fit_result['best_fit']
    model_params = fit_result['model_parameters'][best_fit]
    
    sigma1 = fit_result['fits'][best_fit][0][model_params.index('sigma1')] if 'sigma1' in model_params else np.nan
    sigma2 = fit_result['fits'][best_fit][0][model_params.index('sigma2')] if 'sigma2' in model_params else np.nan
    mu1 = fit_result['fits'][best_fit][0][model_params.index('mu1')] if 'mu1' in model_params else np.nan
    mu2 = fit_result['fits'][best_fit][0][model_params.index('mu2')] if 'mu2' in model_params else np.nan
    C = fit_result['fits'][best_fit][0][model_params.index('C')]

    if not np.isnan(mu1) and not np.isnan(sigma1) and np.abs(sigma1) < 0.5:
        pos1 = mu1
        width1 = min(5.6*np.abs(sigma1), 0.5)
        depth1 = C - fluxes[np.argmin(phases-pos1)]
    else:
        pos1 = np.nan
        width1 = np.nan 
        depth1 = np.nan
    if not np.isnan(mu2) and not np.isnan(sigma2) and np.abs(sigma2) < 0.5:
        pos2 = mu2
        width2 = min(5.6*np.abs(sigma2), 0.5)
        depth2 = C - fluxes[np.argmin(phases-pos2)]
    else:
        pos2 = np.nan
        width2 = np.nan
        depth2 = np.nan
    
    phases_w, fluxes_w, sigmas_w = extend_phasefolded_lc(phases, fluxes, sigmas)
    if not np.isnan(width1) and not np.isnan(width2) and not np.isnan(pos1) and not np.isnan(pos2):
        if np.abs(pos1-pos2) < width1 or np.abs(pos1-pos2) < width2:
            # in case of higly ellipsoidal systems, the eclipse positions aren't detected well
            # and need to be refined
            logger.warning('Poor two-Gaussian fit. Results potentially unreliable!')
            pos1 = phases_w[(phases_w > -0.25) & (phases_w < 0.25)][np.argmin(fluxes_w[(phases_w > -0.25) & (phases_w < 0.25)])]
            pos2 = phases_w[(phases_w > 0.25) & (phases_w < 0.75)][np.argmin(fluxes_w[(phases_w > 0.25) & (phases_w < 0.75)])]
            width1 = 0.5
            width2 = 0.5
                                    
        eclipse_breaks = refine_eclipse_widths(phases_w, fluxes_w, sigmas_w, pos1, pos2, width1, width2)
        if eclipse_breaks is not None:
            width1, width2 = eclipse_breaks[1]-eclipse_breaks[0], eclipse_breaks[3]-eclipse_breaks[2]
        

    if diagnose:
        twogfuncs = {'C': const, 'CE': ce, 'CG': cg, 'CGE': cge, 'CG12': cg12, 'CG12E1': cg12e1, 'CG12E2': cg12e2}
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        plt.plot(phases_w, fluxes_w, 'k.')
        plt.plot(phases_w, twogfuncs[best_fit](phases_w, *fit_result['fits'][best_fit][0]), '-', label=fit_result['best_fit'])
        plt.axvline(x=pos1, c='blue', ls='--', label='primary pos')
        plt.axvline(x=pos2, c='orange', ls='--', label='secondary pos')
        if eclipse_breaks is not None:
            plt.axvline(x=eclipse_breaks[0], c='blue', ls=':')
            plt.axvline(x=eclipse_breaks[1], c='blue', ls=':')
            plt.axvline(x=eclipse_breaks[2], c='orange', ls=':')
            plt.axvline(x=eclipse_breaks[3], c='orange', ls=':')
        else:
            plt.axvline(x=pos1-0.5*width1, c='blue', ls=':')
            plt.axvline(x=pos1+0.5*width1, c='blue', ls=':')
            plt.axvline(x=pos2-0.5*width2, c='orange', ls=':')
            plt.axvline(x=pos2+0.5*width2, c='orange', ls=':')
        plt.legend()
        plt.show()

    return {
        'primary_width': width1,
        'secondary_width': width2,
        'primary_position': pos1,
        'secondary_position': pos2,
        'primary_depth': depth1,
        'secondary_depth': depth2
    }

# ECCENTRICITY AND ARG OF PERIASTRON ESTIMATOR

def f (psi, sep): # used in pf_ecc_psi_w
    return psi - sin(psi) - 2*pi*sep

def df (psi, sep): # used in pf_ecc_psi_w
    return 1 - cos(psi) +1e-6

def ecc_w_from_geometry(sep,pwidth,swidth):

    if np.isnan(sep) or np.isnan(pwidth) or np.isnan(swidth):
        logger.warning('Cannot esimate eccentricty and argument of periastron: incomplete geometry information')
        return 0., pi/2

    psi = newton(func=f, x0=(12*pi*sep)**(1./3), fprime=df, args=(sep,), maxiter=500)
    ecc = sqrt( (0.25*tan(psi-pi)**2+(swidth-pwidth)**2/(swidth+pwidth)**2)/(1+0.25*tan(psi-pi)**2) )
    try:
        w1 = arcsin((pwidth-swidth)/(swidth+pwidth)/ecc)
        w2 = arccos(sqrt(1-ecc**2)/2/ecc*tan(psi-pi))
        w = w2 if w1 >= 0 else 2*pi-w2
    except:
        w = pi/2
    return ecc, w
