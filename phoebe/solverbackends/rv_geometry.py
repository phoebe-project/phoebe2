import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.optimize import newton, minimize, least_squares
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def estimate_period(rvdata, use_sigma=False, component=1):
    from astropy.timeseries import LombScargle
    if use_sigma:
        frequency, power = LombScargle(rvdata[:,0], rvdata[:,1], rvdata[:,2]).autopower(
            minimum_frequency=1./(rvdata[:,0][-1]-rvdata[:,0][0]),
            maximum_frequency=0.5/((rvdata[:,0][-1]-rvdata[:,0][0])/len(rvdata)),
            samples_per_peak=10)
    else:
        frequency, power = LombScargle(rvdata[:,0], rvdata[:,1]).autopower(
            minimum_frequency=1./(rvdata[:,0][-1]-rvdata[:,0][0]),
            maximum_frequency=0.5/(rvdata[:,0][1]-rvdata[:,0][0]),
            samples_per_peak=10)

    return 1./frequency[np.argmax(power)]


def estimate_q(rvdata1, rvdata2):
    rv1_amp = 0.5*(rvdata1[:,1].max()-rvdata1[:,1].min())
    rv2_amp = 0.5*(rvdata2[:,1].max()-rvdata2[:,1].min())
    return rv1_amp/rv2_amp


def estimate_sma(rvdata, period=1*u.d, incl = np.pi/2):
    K = 0.5*(rvdata[:,1].max()-rvdata[:,1].min())
    return K/(2*np.pi/(period.to(u.s)).value*np.sin(incl))


def estimate_vgamma(rvdata):
    K = 0.5*(rvdata[:,1].max()-rvdata[:,1].min())
    return 0.5*((rvdata[:,1].max()-K)+(rvdata[:,1].min()+K))


def estimate_t0_supconj(rvdata, vgamma, period, component=1):
    # TODO: fold with period so there are more points available for interpolation?
    # limit to one period
    times_fold = rvdata[:,0]%period
    s = np.argsort(times_fold)
    times_fold = times_fold[s]
    rvs_fold = rvdata[:,1][s]

    rvs_fold_smooth = savgol_filter(rvs_fold, 51, 3)
    grad = np.gradient(rvs_fold_smooth)

    if component==1:
        t_rv_interp = interp1d(rvs_fold_smooth[grad<=0], times_fold[grad<=0])
    elif component==2:
        t_rv_interp = interp1d(rvs_fold_smooth[grad>=0], times_fold[grad>=0])
    else:
        raise ValueError

    t_vgamma = t_rv_interp(vgamma)

    return t_vgamma

def estimate_some_params(rvdata, period, vgamma, sma, component=1):
    period = estimate_period(rvdata, component=component) if period is None else period
    sma = estimate_sma(rvdata, period = period*u.d) if sma is None else sma
    vgamma = estimate_vgamma(rvdata) if vgamma is None else vgamma

    return period, sma, vgamma

def rv(tanoms, asini=1., q=1., e=0., P=1., per0=0., component=1):
    # asini: km, P:s (passed in days so this will need to be converted first), per0: rad
    # since I'm passing P in days everywhere else, here is the only place I'll use
    # astropy.units to convert to seconds for the computation
    Ps = ((P*u.d).to(u.s)).value
    if component==1:
        const = 2*np.pi*q*asini/(Ps*(1+q)*(1-e**2)**0.5)
    elif component==2:
        const = -2*np.pi*asini/(Ps*(1+q)*(1-e**2)**0.5)
    else:
        raise ValueError('Unrecognized component %i, can only be 1 or 2' % (component))

    tdep = e*np.cos(per0)+np.cos(per0+tanoms)
    return (const*tdep)


def t_supconj_perpass(t0_supconj, period, ecc, per0):
    """
    time shift between superior conjuction and periastron passage
    """
    #     ups_sc = np.pi/2-per0
    ups_sc = per0
    E_sc = 2*np.arctan(np.sqrt((1-ecc)/(1+ecc)) * np.tan(ups_sc/2) )
    M_sc = E_sc - ecc*np.sin(E_sc)
    return t0_supconj - period*(M_sc/2./np.pi)


def ecc_anomaly(x, t, P, t0, e):
    return x-e*np.sin(x) - 2*np.pi/P*(t-t0)


def rv_model(times, t0_supconj, P, per0, ecc, asini, q, vgamma, component=1):

    # provided t0 is t0_supconj, we need to convert it to t0_perpass first
    t0 = t_supconj_perpass(t0_supconj, P, ecc, per0)
    Es = np.zeros(len(times))
    for i,t in enumerate(times):
        Es[i] = newton(ecc_anomaly, 0., args=(t, P, t0, ecc))
    tanoms = 2*np.arctan(((1+ecc)/(1-ecc))**0.5*np.tan(Es/2))
    return rv(tanoms, asini=asini, q=q, e=ecc, P=P, per0=per0, component=component)+vgamma


def loglike(params, rv1data, rv2data, period, q, sma, vgamma, t0):
    logl1 = 0
    logl2 = 0

    per0, ecc = params
    if rv1data is not None:
        rvs1 = rv_model(rv1data[:,0], t0, period, per0, ecc, sma, q, vgamma, component=1)
        logl1 = -0.5*np.sum((rv1data[:,1]-rvs1)**2/(rv1data[:,2])**2)

    if rv2data is not None:
        rvs2 = rv_model(rv2data[:,0], t0, period, per0, ecc, sma, q, vgamma, component=2)
        logl2 = -0.5*np.sum((rv2data[:,1]-rvs2)**2/(rv2data[:,2])**2)
#     print(logl1+logl2)
    return logl1+logl2


def estimate_rv_parameters(rv1data=None, rv2data=None,
                           period=None, t0=None, q=None, vgamma=None, sma=None, ecc=None, per0=None):

    if rv1data is None and rv2data is None:
        raise ValueError('Both rv1 and rv2 data cannot be None. Please provide at least one.')

    if rv1data is not None and rv2data is None:
        q = 1 if q is None else q
        period, sma, vgamma = estimate_some_params(rv1data, period, vgamma, sma, component=1)
        sma = 2*sma
        t0 = estimate_t0_supconj(rv1data, vgamma, period, component=1) if t0 is None else t0

    if rv2data is not None and rv1data is None:
        q = 1 if q is None else q
        period, sma, vgamma = estimate_some_params(rv2data, period, vgamma, sma, component=1)
        sma = 2*sma
        t0 = estimate_t0_supconj(rv2data, vgamma, period, component=2) if t0 is None else t0

    if rv1data is not None and rv2data is not None:
        q = estimate_q(rv1data, rv2data) if q is None else q
        period1, sma1, vgamma1 = estimate_some_params(rv1data, period, vgamma, sma, component=1)
        period2, sma2, vgamma2 = estimate_some_params(rv2data, period, vgamma, sma, component=2)

        period = np.mean([period1, period2])
        sma = sma1+sma2
        vgamma = np.mean([vgamma1, vgamma2])

        t01 = estimate_t0_supconj(rv1data, vgamma, period, component=1) if t0 is None else t0
        t02 = estimate_t0_supconj(rv2data, vgamma, period, component=2) if t0 is None else t0
        t0 = np.mean([t01, t02])

    # set initial values for ecc and per0
    ecc = 0 if ecc is None else ecc
    per0 = 0 if per0 is None else per0

    init_params = [per0, ecc]
    result = least_squares(loglike, x0=init_params, ftol=1e-8, xtol=1e-8,
          #bounds = ((times.min(), 0, 0., rvs.min()),(times.min()+period, 2*np.pi, 0.9, rvs.max())),
          bounds = ((0.,0.), (2*np.pi, 0.9)),
          kwargs={'rv1data':rv1data, 'rv2data':rv2data,
                  'period': period, 'q':q, 'sma': sma1+sma2,
                 'vgamma': vgamma, 't0':t0})

    return {'period':period, 't0_supconj':t0, 'q':q, 'asini':sma,
            'vgamma':vgamma, 'ecc':result.x[1], 'per0':result.x[0],
            'rv1_analytic': rv_model(rv1data[:,0], t0, period, result.x[0], result.x[1], sma, q, vgamma, component=1),
            'rv2_analytic': rv_model(rv2data[:,0], t0, period, result.x[0], result.x[1], sma, q, vgamma, component=2)}
