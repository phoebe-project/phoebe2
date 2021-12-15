import numpy as np
from scipy.special import binom as binomial

__version__ = '0.1.0'


def tabulate(args):
    """
    Convert any combination of scalars and arrays into a table.

    Interpolators need fixed size arrays to do their work, but it is not
    convenient to have to pass arrays for single values. For example, to
    run the interpolator over one axis while keeping the other axes constant,
    the calling function can pass an array and a set of scalars. This
    function will then convert the scalars to arrays and tabulate all arrays
    as needed for the interpolator.

    Parameters
    ----------
    * `args` (list):
        a list of scalars and arrays to be tabulated

    Returns
    -------
    * (ndarray) tabulated N-d array
    """

    args = [np.atleast_1d(arg) for arg in args]
    lens = np.array([len(arg) for arg in args], dtype=int)
    if len(np.unique(lens)) == 1:
        return np.vstack(args).T
    max_length = lens.max()
    for i, arg in enumerate(args):
        if len(arg) == 1:
            args[i] = np.full(max_length, arg[0])
            continue
        if len(arg) != max_length:
            raise ValueError(f'cannot tabulate arrays of varying lengths {lens}.')
    return np.vstack(args).T


def ndpolate(x, lo, hi, fv, copy_data=False):
    """
    N-dimensional linear interpolation.

    Linear interpolation or extrapolation in N dimensions.

    Given a sequence of axes, interpolation successively reduces the number of
    dimensions in which it interpolates. It starts with the corners of the
    N-dimensional hypercube to obtain interpolated values in the
    (N-1)-dimensional hyperplane while keeping the last axis constant. It then
    removes the last axis and forms an (N-1)-dimensional hypercube from the
    interpolated vertices. The process is then restarted and continued until
    we obtain the interpolated value in the point of interest.

    Note that the choice of interpolating along the last axis is both
    arbitrary (we could just as easily choose any other axis to start with)
    and general (we could pivot the axes if another interpolation sequence is
    desired). For as long as the parameter space is sufficiently linear, the
    choice of axis sequence is not too important, but any local non-linearity
    will cause the sequence to matter.

    The algorithm takes a vector (or an array of vectors) of interest `x`
    (open circle), an N-dimensional vector of lower vertex axis values `lo`,
    an N-dimensional vector of upper vertex axis values `hi`, and an array of
    2^N function values `fv` sorted by vertices in the following sequence:

         | f(x1_lo, x2_lo, ..., xN_lo) |
         | f(x1_hi, x2_lo, ..., xN_lo) |
         | f(x1_lo, x2_hi, ..., xN_lo) |
    fv = | f(x1_hi, x2_hi, ..., xN_lo) |
         |        . . . . . . .        |
         | f(x1_lo, x2_hi, ..., xN_hi) |
         | f(x1_hi, x2_hi, ..., xN_hi) |

    where xk are the axis values for the k-th axis. Interpolation proceeds
    from the last axis to the first, and array `fv` is modified in the
    process.

    Parameters
    ----------
    * `x` (array): vector of interest
    * `lo` (array): N-dimensional vector of lower knot values
    * `hi` (array): N-dimensional vector of upper knot values
    * `fv` (array): (2^N)-dimensional vector of function values at knots. Note
        that the `fv` array is modified so make sure you pass a copy if you
        need to reuse it, or set `copy_data`=True if you want to work on a
        copy.
    * `copy_data` (bool, optional, default=False): switch to control whether a
        local copy of fv should be made before it is modified.

    Returns
    -------
    * (float) interpolated value at `x`
    """

    N = len(lo)
    powers = [2**k for k in range(N+1)]

    n = np.empty((powers[N], N))

    lfv = fv.copy() if copy_data else fv

    for i in range(N):
        for j in range(powers[N]):
            n[j, i] = lo[i] + (int(j/powers[i]) % 2) * (hi[i]-lo[i])

    for i in range(N):
        for j in range(powers[N-i-1]):
            lfv[j] += (x[N-i-1]-n[j, N-i-1])/(n[j+powers[N-i-1], N-i-1]-n[j, N-i-1])*(lfv[j+powers[N-i-1]]-lfv[j])

    return lfv[0]


def interpolate_all_directions(entry, axes, grid):
    """
    Interpolates the value across all defined directions.

    If all bounding vertex values are defined, there are (N over D)
    combinations to interpolate in D-dimensional subspace. In 3D, there is 1
    combination in 3D, 3 combinations in 2D, and 3 combinations in 1D. If
    there are any NaN values in the grid, those values will fail to
    interpolate.

    If the grid were truly linear, then it would not matter along which
    direction we interpolate -- all directions would yield the same answer. It
    is however uncommon that the parameter space is truly linear, so each
    interpolated direction will yield a different result. Because of that, we
    can join individual values either by taking a simple mean, or by first
    averaging them per dimension of the subspace (i.e., over (N over D)
    combinations), and then taking the mean. The function returns an array of
    all interpolants, so the calling function can apply any weighting scheme
    that is suitable for the problem at hand.

    The order of directions is determined by the `mask` parameter. It flags
    the axes that are "in use". For example, for a 3D grid, the sequence is:

    [1,1,1], [0,1,1], [1,0,1], [1,1,0], [0,0,1], [0,1,0], [1,0,0].

    Parameters
    ----------
    * `entry` (tuple): a point of interest
    * `axes` (tuple): tuple of interpolation axes
    * `grid` (ndarray): N-D array of function values that enclose the point of
        interest

    Returns
    -------
    * (array) interpolated values for all directions
    """

    N = len(entry)
    interpolants = []

    for D in range(N, 0, -1):  # sub-dimensions
        for d in range(int(binomial(N, D))):  # combinations per sub-dimension
            slc = [slice(max(0, entry[k]-1), min(entry[k]+2, len(axes[k])), 2) for k in range(N)]
            mask = np.ones(N, dtype=bool)

            for k in range(N-D):  # projected axes
                slc[(d+k) % N] = slice(entry[(d+k) % N], entry[(d+k) % N]+1)
                mask[(d+k) % N] = False

            fv = grid[tuple(slc)].reshape(-1, 1)
            if len(fv) != 2**mask.sum():  # missing vertices, cannot calculate
                continue

            x = np.array([axes[i][entry[i]] for i in range(N)])[mask]
            lo = np.array([axes[i][max(0, entry[i]-1)] for i in range(N)])[mask]
            hi = np.array([axes[i][min(entry[i]+1, len(axes[i])-1)] for i in range(N)])[mask]
            fv = grid[tuple(slc)].reshape(-1, 1)

            interpolants.append(ndpolate(x, lo, hi, fv, copy_data=True))

    return np.array(interpolants)


def map_to_cube(v, axes, intervals):
    """
    Non-conformal mapping of the original space to an N-dimensional cube.

    Axes that span the parameter space are frequently given in physical units,
    so it is impractical to interpolate or extrapolate in relative terms: for
    example, 1% along one axis might exact to a small change while 1% along
    another could be appreciable. To rectify that, we can map the parameter
    space spun by the original axes to an N-dimensional hypercube by rescaling
    the axes and shifting them appropriately. In addition, this function
    allows non-conformal stretching of the parameter space by providing the
    unit interval size at the beginning and at the end of each axis.

    Parameters
    ----------
    * `v` (array):
        vector in old coordinates
    * `axes` (tuple of arrays):
        a list of original axes
    * `intervals` (tuple of 2-D arrays):
        an array of step sizes at the beginning and the end of the new axes.
    
    Returns
    -------
    * (array) vector in new coordinates
    """

    retval = []
    for k in range(len(v)):
        ranges = (axes[k][0], axes[k][-1])
        delta_k = intervals[k][0] + (v[k]-ranges[k][0])/(ranges[k][1]-ranges[k][0])*(intervals[k][1]-intervals[k][0])
        retval.append((v[k]-ranges[k][0])/delta_k)

    return tuple(retval)
