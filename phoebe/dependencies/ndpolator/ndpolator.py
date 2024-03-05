import numpy as np
from scipy.special import binom as binomial
from enum import IntEnum

import cndpolator

__version__ = '0.1.0'


class ExtrapolationMethod(IntEnum):
    NONE = 0
    NEAREST = 1
    LINEAR = 2


class AxisFlag(IntEnum):
    SPANNING_AXIS = 1
    ADDITIONAL_AXIS = 2


class Cndpolator():
    """
    This class implements interpolation and extrapolation in n dimensions.
    """
    def __init__(self, axes):
        """
        Instantiates a Cndpolator class. The class relies on `axes` to span
        the interpolation hypercubes. Only basic (spanning) axes should
        be passed here. The axes and the nanmask are stored, and a list of
        defined nodes (nodes with non-nan values) and a list of fully defined
        hypercubes (hypercubes with all non-nan vertices) is also stored.

        Parameters
        ----------
        axes : tuple of ndarrays
            Axes that span the atmosphere grid. Only the required (spanning)
            axes should be included here; any additional axes should be
            registered separately.
        """

        self.axes = axes
        self.table = dict()

    def __repr__(self) -> str:
        return f'<Ndpolator N={len(self.axes)}, {len(self.table)} tables>'

    def __str__(self) -> str:
        return f'<Ndpolator N={len(self.axes)}, {len(self.table)} tables>'

    @property
    def tables(self):
        """
        Prints a list of tables attached to the ndpolator.

        Returns
        -------
        list of strings
            table names (references) attached to the ndpolator
        """
        return list(self.table.keys())

    def register(self, table, adtl_axes, grid):
        if not isinstance(table, str):
            raise ValueError('parameter `table` must be a string')

        self.table[table] = [adtl_axes, np.ascontiguousarray(grid), None]

    def find_indices(self, table, query_pts):
        adtl_axes = self.table[table][0]
        axes = self.axes if adtl_axes is None else self.axes + adtl_axes
        indices, flags, normed_query_pts = cndpolator.find(axes, query_pts)
        return indices, flags, normed_query_pts

    def find_hypercubes(self, table, indices, flags, adtl_axes=None):
        axes = self.axes if adtl_axes is None else self.axes + adtl_axes
        grid = self.table[table][1]
        hypercubes = cndpolator.hypercubes(indices, axes, flags, grid)
        return hypercubes

    def ndpolate(self, table, query_pts, extrapolation_method=0):
        if extrapolation_method == 'none':
            extrapolation_method = 0
        elif extrapolation_method == 'nearest':
            extrapolation_method = 1
        elif extrapolation_method == 'linear':
            extrapolation_method = 2
        else:
            raise ValueError(f"extrapolation_method={extrapolation_method} is not valid; it must be one of ['none', 'nearest', 'linear'].")

        capsule = self.table[table][2]
        if capsule:
            interps = cndpolator.ndpolate(capsule=capsule, query_pts=query_pts, nbasic=len(self.axes), extrapolation_method=extrapolation_method)
        else:
            attached_axes = self.table[table][0]
            grid = self.table[table][1]
            axes = self.axes if attached_axes is None else self.axes + attached_axes

            interps, capsule = cndpolator.ndpolate(query_pts=query_pts, axes=axes, grid=grid, nbasic=len(self.axes), extrapolation_method=extrapolation_method)
            self.table[table][2] = capsule

        return interps


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

    The algorithm takes a vector (or an array of vectors) of interest `x`, an
    N-dimensional vector of lower vertex axis values `lo`, an N-dimensional
    vector of upper vertex axis values `hi`, and an array of 2^N function
    values `fv` sorted by vertices in the following sequence:

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


def impute_grid(axes, grid, weighting='none'):
    """
    Imputes missing values in the grid.

    The function traverses the passed `grid` and finds all `nan`s. It then
    interpolates the missing values along all directions, calculates a simple
    mean and imputes the missing value in-place (i.e., it modifies the passed
    grid).

    Parameters
    ----------
    * `axes` (tuple of arrays): a list of axes
    * `grid` (ndarray): N-D grid to be imputed
    * `weighting` (string): weighting method. Only 'none' is currently
      implemented, but other weighting schemes should be added.
    """

    if weighting != 'none':
        raise NotImplementedError(f'weighting={weighting} is currently not supported.')

    nantable = np.argwhere(np.isnan(grid[..., 0]))
    for entry in nantable:
        interps = interpolate_all_directions(entry=entry, axes=axes, grid=grid)
        if np.all(np.isnan(interps)):
            continue
        interps = interps[~np.isnan(interps)].mean()
        grid[tuple(entry)][0] = interps
