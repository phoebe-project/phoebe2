import numpy as np
from astropy import units as u
from scipy.optimize import newton, minimize, least_squares
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.version import version as _scipy_version
from phoebe.constraints.builtin import t0_supconj_to_perpass
from copy import deepcopy
from distutils.version import LooseVersion

# if os.getenv('PHOEBE_ENABLE_PLOTTING', 'TRUE').upper() == 'TRUE':
#     try:
#         import matplotlib.pyplot as plt
#     except (ImportError, TypeError):
#         _use_mpl = False
#     else:
#         _use_mpl = True
# else:
#     _use_mpl = False


def smooth_rv(rvdata):
    if rvdata is None:
        return None

    win_len = int(len(rvdata)/5)
    win_len = int(win_len + 1) if int(win_len)%2 == 0 else int(win_len)
    win_len = 5 if win_len <= 3 else win_len
    poly_ord = 3
    rv_smooth = savgol_filter(rvdata[:,1], window_length=win_len, polyorder=poly_ord)
 
    return np.array([rvdata[:,0], rv_smooth, rvdata[:,2]]).T


def estimate_q(rv1data, rv2data, vgamma=None):
    rv1max = max(rv1data[:,1])
    rv2min = rv2data[:,1][rv1data[:,1]==rv1max]
#     rv2max = max(rv2data[:,1])
    if vgamma is None:
        return np.abs(rv1max/rv2min)
    else:
        return (rv1max-vgamma)/(-rv2min+vgamma)


def estimate_vgamma(rv1data, rv2data, q=1.):
    return np.mean(rv1data[:,1]+q*rv2data[:,1])/(1+q)


def estimate_q_vgamma(rv1data, rv2data, maxiter=10):
    if rv2data is None:
        rv1_flipped = np.array([rv1data[:,0],
                                -rv1data[:,1] + rv1data[:,1].max() + rv1data[:,1].min(),
                                rv1data[:,2]]).T
        return np.nan, estimate_vgamma(rv1data, rv1_flipped, q=1.)

    if rv1data is None:
        rv2_flipped = np.array([rv2data[:,0],
                                -rv2data[:,1] + rv2data[:,1].max() + rv2data[:,1].min(),
                                rv2data[:,2]]).T
        return np.nan, estimate_vgamma(rv2data, rv2_flipped, q=1.)


    q_prev = estimate_q(rv1data, rv2data, vgamma=None)
    vgamma_prev = estimate_vgamma(rv1data, rv2data, q_prev)

    for i in range(maxiter):
        q_est = estimate_q(rv1data, rv2data, vgamma=vgamma_prev)
        vgamma_est = estimate_vgamma(rv1data, rv2data, q_est)

        if np.abs((q_est - q_prev)/q_est) < 1e-2 and np.abs((vgamma_est -  vgamma_prev)/vgamma_est) < 1e-2:
            break

        q_prev = q_est
        vgamma_prev = vgamma_est

    return q_est[0], vgamma_est[0]


def estimate_asini(rv1data, rv2data, period = 1*u.d, vgamma = 0., ecc=0.):
    period  = (period.to(u.s)).value

    K1 = 0.5*(max(rv1data[:,1]-vgamma)-min(rv1data[:,1]-vgamma)) if rv1data is not None else np.nan
    asini1 = K1*period*(1-ecc**2)**0.5/(2*np.pi)
    K2 = 0.5*(max(rv2data[:,1]-vgamma)-min(rv2data[:,1]-vgamma)) if rv2data is not None else np.nan
    asini2 = K2*period*(1-ecc**2)**0.5/(2*np.pi)

    return [asini1, asini2]


def estimate_phase_supconj(rv1data, rv2data, vgamma):

    if rv1data is not None:
        grad1 = np.gradient(rv1data[:,1]) if rv1data is not None else np.nan
        ph_rv1_interp = interp1d(rv1data[:,1][grad1<=0], rv1data[:,0][grad1<=0])
        ph_vgamma_1 = ph_rv1_interp(vgamma)
    else:
        ph_vgamma_1 = np.nan

    if rv2data is not None:
        grad2 = np.gradient(rv2data[:,1]) if rv2data is not None else np.nan
        ph_rv2_interp = interp1d(rv2data[:,1][grad2>=0], rv2data[:,0][grad2>=0])
        ph_vgamma_2 = ph_rv2_interp(vgamma)
    else:
        ph_vgamma_2 = np.nan

    return np.nanmean([ph_vgamma_1, ph_vgamma_2])


def ecc_anomaly(x, phases, ph0, ecc):
    return x-ecc*np.sin(x) - 2*np.pi*(phases-ph0)


def rv_model(phases, P, per0, ecc, asini, vgamma, ph_supconj, component=1):

    ph0 = t0_supconj_to_perpass(ph_supconj, 1., ecc, per0, 0., 0., 0.)
    # deepcopy is needed for scipy < 1.2.2 because of this bug: https://github.com/scipy/scipy/issues/9964
    Es = newton(ecc_anomaly,
                deepcopy(phases) if LooseVersion(_scipy_version) < LooseVersion("1.2.2") else phases,
                args=(phases, ph0*np.ones_like(phases), ecc*np.ones_like(phases)))

    thetas = 2*np.arctan(((1+ecc)/(1-ecc))**0.5*np.tan(Es/2))
    P_s = ((P*u.d).to(u.s)).value
    if component==1:
        const = 2*np.pi*asini[0]/(P_s*(1-ecc**2)**0.5)
    elif component==2:
        const = -2*np.pi*asini[1]/(P_s*(1-ecc**2)**0.5)
    else:
        raise ValueError('Unrecognized component %i, can only be 1 or 2' % (component))

    tdep = ecc*np.cos(per0)+np.cos(per0+thetas)
    return (const*tdep) + vgamma


def loglike(params, rv1data, rv2data, asini, vgamma, ph_supconj):
    logl1 = 0
    logl2 = 0

    ecc, per0 = params
    period = 1. # because phase-folded rv
    if rv1data is not None:
        rvs1 = rv_model(rv1data[:,0], period, per0, ecc, asini, vgamma, ph_supconj,component=1)
        logl1 = -0.5*np.sum((rv1data[:,1]-rvs1)**2/(rv1data[:,2])**2)

    if rv2data is not None:
        rvs2 = rv_model(rv2data[:,0], period, per0, ecc, asini, vgamma, ph_supconj, component=2)
        logl2 = -0.5*np.sum((rv2data[:,1]-rvs2)**2/(rv2data[:,2])**2)
#     print(logl1+logl2)
    return logl1+logl2


def estimate_rv_parameters(rv1data=None, rv2data=None,
                           q=None, vgamma=None, asini=None, ecc=None, per0=None, maxiter=10):

    rv1_smooth = smooth_rv(rv1data) if rv1data is not None else rv1data
    rv2_smooth = smooth_rv(rv2data) if rv2data is not None else rv2data

    q, vgamma = estimate_q_vgamma(rv1_smooth, rv2_smooth)
    asinis = estimate_asini(rv1_smooth, rv2_smooth, period = 1.*u.d, vgamma = vgamma, ecc=0.)
    ph_supconj = estimate_phase_supconj(rv1_smooth, rv2_smooth, vgamma)
    # set initial values for ecc and per0
    ecc_inits = [0., 0.4]
    per0_inits = [0., np.pi/2, np.pi]

    loglikes = np.zeros((2,3))
    results = np.zeros((2,3,2))
    for i,ecc in enumerate(ecc_inits):
        for j,per0 in enumerate(per0_inits):
            init_params = [ecc, per0]
            for k in range(maxiter):
                result = least_squares(loglike, x0=init_params, ftol=1e-8, xtol=1e-8,
                    #bounds = ((times.min(), 0, 0., rvs.min()),(times.min()+period, 2*np.pi, 0.9, rvs.max())),
                    bounds = ((0.,0.), (0.9, 2*np.pi)),
                    kwargs={'rv1data':rv1data, 'rv2data':rv2data,
                            'asini': asinis,
                            'vgamma': vgamma, 'ph_supconj':ph_supconj})
                asinis = estimate_asini(rv1data, rv2data, period = 1.*u.d, vgamma = vgamma, ecc=result.x[0])
                if np.abs((result.x[0] - init_params[0])/result.x[0]) < 1e-2 and np.abs((result.x[1] - init_params[1])/result.x[1]) < 1e-2:
                    break
                init_params = result.x

            loglikes[i,j] = loglike(result.x, rv1data, rv2data, asinis, vgamma, ph_supconj)
            results[i,j] = result.x

    [ecc, per0] = results.reshape(6,2)[np.argmax(loglikes.reshape(6))]
    return {'q':q, 'asini': np.array(asinis),
            'vgamma':vgamma, 'ecc':ecc, 'per0':per0, 'ph_supconj': ph_supconj}
            # 'rv1_analytic': rv_model(rv1data[:,0], t0, period, result.x[0], result.x[1], asini*period, q, vgamma, component=1),
            # 'rv2_analytic': rv_model(rv2data[:,0], t0, period, result.x[0], result.x[1], asini*period, q, vgamma, component=2)}
