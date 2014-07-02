"""
Periodogram calculations
"""
import logging
import numpy as np
from numpy import pi

logger = logging.getLogger('UTILS.PERGRAM')

def DFT(time, x, freq):
    """
    Compute the Discrete Fourier Transform.
    
    you could try freq[:,None] for a frequency array, as long as it's
    not too big.
    """
    N = len(time)
    return np.sum(x*np.exp(1j*2*np.pi*freq*time),axis=-1)

def DFTpower(time, signal, f0=None, fn=None, df=None, freqs=None, full_output=False):

    """
    Computes the modulus square of the fourier transform. 
    Unit: square of the unit of signal. Time points need not be equidistant.
    The normalisation is such that a signal A*sin(2*pi*nu_0*t)
    gives power A^2 at nu=nu_0
    
    @param time: time points [0..Ntime-1] 
    @type time: ndarray
    @param signal: signal [0..Ntime-1]
    @type signal: ndarray
    @param f0: the power is computed for the frequencies
                      freq = arange(startfreq,stopfreq,stepfreq)
    @type f0: float
    @param fn: see startfreq
    @type fn: float
    @param df: see startfreq
    @type df: float
    @return: power spectrum of the signal
    @rtype: ndarray 
    """
    time = np.asarray(time)
    signal = np.asarray(signal)
    if f0 is None:
        f0 = 0.
    if fn is None:
        fn = 0.5/np.median(np.diff(time))
    if df is None:
        df = 0.1/time.ptp()
    
    Ntime = len(time)
    if freqs is None and fn>f0:
        freqs = np.arange(f0,fn,df)
        Nfreq = int(np.ceil((fn-f0)/df))
    elif freqs is None:
        freqs = np.array([f0])
        df = 0
        Nfreq = 2
  
    A = np.exp(1j*2.*pi*f0*time) * signal
    B = np.exp(1j*2.*pi*df*time)
    ft = np.zeros(Nfreq, complex) 
    ft[0] = A.sum()
    
    for k in range(1,Nfreq):
        A *= B
        ft[k] = np.sum(A)
    if df==0:
        freqs = freqs[0]
        ft = ft[-1]
        
    if full_output:
        return freqs,ft
    else:
        return freqs,(ft.real**2 + ft.imag**2) * 4.0 / Ntime**2