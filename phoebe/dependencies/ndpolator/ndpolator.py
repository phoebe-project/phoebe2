import numpy as np

__version__ = '0.1.0'

def ndpolate(x, lo, hi, fv, copy_data=False):
    """
    @x: vector of interest
    @lo: N-dimensional vector of lower knot values
    @hi: N-dimensional vector of upper knot values
    @fv: (2^N)-dimensional vector of function values at knots
    @copy_data: boolean switch to control whether a local copy of fv
    should be made before it is modified.

    Linear interpolation or extrapolation in N dimensions. The fv array
    is modified so make sure you pass a copy if you need to reuse it,
    or set copy_data=True if you want to work on a copy.
    """

    N = len(lo)
    powers = [2**k for k in range(N+1)]

    n = np.empty((powers[N], N))

    lfv = fv.copy() if copy_data else fv

    for i in range(N):
        for j in range(powers[N]):
            n[j,i] = lo[i] + (int(j/powers[i]) % 2) * (hi[i]-lo[i])
    
    for i in range(N):
        for j in range(powers[N-i-1]):
            lfv[j] += (x[N-i-1]-n[j,N-i-1])/(n[j+powers[N-i-1],N-i-1]-n[j,N-i-1])*(lfv[j+powers[N-i-1]]-lfv[j])

    return lfv[0]
