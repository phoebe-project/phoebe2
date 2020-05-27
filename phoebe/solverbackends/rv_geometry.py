import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.optimize import newton, minimize, least_squares
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def smooth_rv(rvdata):
    if rvdata is None:
        return None
    
    win_len = int(len(rvdata))/5
    win_len = int(win_len + 1) if win_len%2 == 0 else int(win_len)
    win_len = 5 if win_len <= 3 else win_len
    poly_ord = 3
    rv_smooth = savgol_filter(rvdata[:,1], window_length=win_len, polyorder=poly_ord)

    return np.array([rvdata[:,0], rv_smooth, rvdata[:,2]]).T


def estimate_q(rvdata1, rvdata2):
    rv1_amp = 0.5*(rvdata1[:,1].max()-rvdata1[:,1].min())
    rv2_amp = 0.5*(rvdata2[:,1].max()-rvdata2[:,1].min())
    return rv1_amp/rv2_amp


def estimate_asini(rvdata, period=1*u.d):
    K = 0.5*(rvdata[:,1].max()-rvdata[:,1].min())
    return K/(2*np.pi/(period.to(u.s)).value)


def estimate_vgamma(rvdata):
    K = 0.5*(rvdata[:,1].max()-rvdata[:,1].min())
    return 0.5*((rvdata[:,1].max()-K)+(rvdata[:,1].min()+K))


def estimate_phase_supconj(rvdata, vgamma, component=1):

    grad = np.gradient(rvdata[:,1])

    if component==1:
        ph_rv_interp = interp1d(rvdata[:,1][grad<=0], rvdata[:,0][grad<=0])
    elif component==2:
        ph_rv_interp = interp1d(rvdata[:,1][grad>=0], rvdata[:,0][grad>=0])
    else:
        raise ValueError

    ph_vgamma = ph_rv_interp(vgamma)

    return ph_vgamma



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


def loglike(params, rv1data, rv2data, q, asini, vgamma, t0):
    logl1 = 0
    logl2 = 0

    per0, ecc = params
    period = 1. # because phase-folded rv
    if rv1data is not None:
        rvs1 = rv_model(rv1data[:,0], t0, period, per0, ecc, asini, q, vgamma, component=1)
        logl1 = -0.5*np.sum((rv1data[:,1]-rvs1)**2/(rv1data[:,2])**2)

    if rv2data is not None:
        rvs2 = rv_model(rv2data[:,0], t0, period, per0, ecc, asini, q, vgamma, component=2)
        logl2 = -0.5*np.sum((rv2data[:,1]-rvs2)**2/(rv2data[:,2])**2)
#     print(logl1+logl2)
    return logl1+logl2


def estimate_rv_parameters(rv1data=None, rv2data=None,
                           period =1., t0=None, q=None, vgamma=None, asini=None, ecc=None, per0=None):

    rv1_smooth = rv1data
    rv2_smooth = rv2data

    if rv1data is None and rv2data is None:
        raise ValueError('Both rv1 and rv2 data cannot be None. Please provide at least one.')

    if rv1data is not None and rv2data is None:
        rv1_smooth = smooth_rv(rv1data)
        q = 1 if q is None else q
        # period, sma, vgamma = estimate_some_params(rv1data, period, vgamma, sma, component=1)
        asini = estimate_asini(rv1_smooth) 
        asini = 2*asini if asini is None else asini
        vgamma = estimate_vgamma(rv1_smooth) if vgamma is None else vgamma
        t0 = estimate_phase_supconj(rv1_smooth, vgamma, component=1) if t0 is None else t0

    if rv2data is not None and rv1data is None:
        rv2_smooth = smooth_rv(rv2data)
        q = 1 if q is None else q
        # period, sma, vgamma = estimate_some_params(rv1data, period, vgamma, sma, component=1)
        asini = estimate_asini(rv2_smooth) 
        asini = 2*asini if asini is None else asini
        vgamma = estimate_vgamma(rv2_smooth) if vgamma is None else vgamma
        t0 = estimate_phase_supconj(rv2_smooth, vgamma, component=2) if t0 is None else t0

    if rv1data is not None and rv2data is not None:
        rv1_smooth = smooth_rv(rv1data)
        rv2_smooth = smooth_rv(rv2data)

        q = estimate_q(rv1_smooth, rv2_smooth) if q is None else q
        if asini is None:
            asini1 = estimate_asini(rv1_smooth)
            asini2 = estimate_asini(rv2_smooth)
            asini = asini1+asini2
        if vgamma is None:
            vgamma1 = estimate_vgamma(rv1_smooth)
            vgamma2 = estimate_vgamma(rv2_smooth)
            vgamma = np.mean([vgamma1, vgamma2])

        if t0 is None:
            t01 = estimate_phase_supconj(rv1_smooth, vgamma, component=1)
            t02 = estimate_phase_supconj(rv2_smooth, vgamma, component=2)
            t0 = np.mean([t01, t02])

    # set initial values for ecc and per0
    ecc = 0 if ecc is None else ecc
    per0 = 0 if per0 is None else per0

    init_params = [per0, ecc]
    result = least_squares(loglike, x0=init_params, ftol=1e-8, xtol=1e-8,
          #bounds = ((times.min(), 0, 0., rvs.min()),(times.min()+period, 2*np.pi, 0.9, rvs.max())),
          bounds = ((0.,0.), (2*np.pi, 0.9)),
          kwargs={'rv1data':rv1data, 'rv2data':rv2data,
                  'q':q, 'asini': asini,
                 'vgamma': vgamma, 't0':t0})


    return {'t0_supconj':t0*period, 'q':q, 'asini':asini*period,
            'vgamma':vgamma, 'ecc':result.x[1], 'per0':result.x[0]}
            # 'rv1_analytic': rv_model(rv1data[:,0], t0, period, result.x[0], result.x[1], asini*period, q, vgamma, component=1),
            # 'rv2_analytic': rv_model(rv2data[:,0], t0, period, result.x[0], result.x[1], asini*period, q, vgamma, component=2)}
