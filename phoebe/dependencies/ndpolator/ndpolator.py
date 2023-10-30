import numpy as np
from scipy.special import binom as binomial
from scipy.spatial import cKDTree
from itertools import product

import cndpolator

__version__ = '0.1.0'

class Cndpolator():
    def __init__(self, axes, grid):
        self.axes = axes
        self.grid = grid

        self.neighbors = np.argwhere(~np.isnan(grid[...,0]))

    def find_indices(self, query_pts):
        return cndpolator.find(self.axes, query_pts)
    
    def find_hypercubes(self, indices, grid):
        return cndpolator.hypercubes(indices, grid)
    
    def interp(self, query_pts, indices=None, hypercubes=None, raise_on_nans=False, return_nanmask=False, extrapolation_method=None):
        # if extrapolation_method is not None or raise_on_nans == True or return_nanmask == True:
        #     print(f'{extrapolation_method=} {raise_on_nans=} {return_nanmask=}')
            # raise ValueError('Almost there, implementing this as we speak.')
        if indices is None:
            indices = self.find_indices(query_pts)
        if hypercubes is None:
            hypercubes = self.find_hypercubes(indices, self.grid)
        
        return cndpolator.ndpolate(query_pts, indices, self.axes, hypercubes)


class Ndpolator():
    def __init__(self, axes, grid, impute=False):
        self.axes = axes
        self.grid = grid

        if impute:
            raise NotImplementedError('consider if this is worth implementing here.')

        self.indices = np.argwhere(~np.isnan(grid[...,0]))
        non_nan_vertices = np.array([ [self.axes[i][self.indices[k][i]] for i in range(len(self.axes))] for k in range(len(self.indices))])
        self.nntree = cKDTree(non_nan_vertices, copy_data=True)
        ranges = (range(len(ax)-1) for ax in axes)
        slices = [tuple([slice(elem, elem+2) for elem in x]) for x in product(*ranges)]
        self.ics = np.array([s.start for slc in slices for s in slc if ~np.any(np.isnan(grid[slc]))], dtype=int).reshape(-1, len(axes))  # bottleneck

    def tabulate(self, arrs):
        arrs = [np.atleast_1d(arr) for arr in arrs]
        lens = np.array([len(arr) for arr in arrs], dtype=int)
        if len(np.unique(lens)) == 1:
            return np.vstack(arrs).T
        max_length = lens.max()
        for i, arr in enumerate(arrs):
            if len(arr) == 1:
                arrs[i] = np.full(max_length, arr[0])
                continue
            if len(arr) != max_length:
                raise ValueError(f'cannot tabulate arrays of varying lengths {lens}.')
        return np.vstack(arrs).T

    def ndpolate(self, x, lo, hi, fv, copy_data=True):
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
        
    def interp(self, req, raise_on_nans=True, return_nanmask=False, extrapolation_method='none'):
        los = np.array([np.searchsorted(self.axes[k], req[:,k], 'left')-1 for k in range(req.shape[1])], dtype=int).T
        his = los+2

        vals = np.empty(shape=(req.shape[0], self.grid.shape[-1]))
        for i, (v, ilo, ihi) in enumerate(zip(req, los, his)):
            slc = tuple([slice(l, h) for l, h in zip(ilo, ihi)])
            try:
                subaxes = np.array([self.axes[k][slc[k]] for k in range(req.shape[1])])
                subgrid = self.grid[slc]
                lo, hi = subaxes[:,0], subaxes[:,1]
                # print(f'subaxes={subaxes}, subgrid shape={subgrid.shape}, subgrid={subgrid}')
            except:
                vals[i] = np.nan
                continue
            pivot_indices = np.roll(np.arange(len(subaxes), -1, -1), -1)
            fv = subgrid.transpose(*pivot_indices).reshape((2**req.shape[1], self.grid.shape[-1]))
            # fv = subgrid.reshape((2**req.shape[1], self.grid.shape[-1]))
            # print(f'fv.shape={fv.shape}\nfv={fv}')
            vals[i] = self.ndpolate(v, lo, hi, fv)

        nanmask = np.isnan(vals[:,0])
        if ~np.any(nanmask):
            return (vals, nanmask) if return_nanmask else vals

        nan_indices = np.argwhere(nanmask).flatten()

        # print(nanmask.shape)
        # print(np.argwhere(nanmask))
        # print(nan_indices.shape)
        # print(f'nan_indices: {nan_indices}')

        if extrapolation_method == 'none' and raise_on_nans:
            raise ValueError(f'params out of bounds: {req[nan_indices]}')
        elif extrapolation_method == 'nearest':
            for k in nan_indices:
                # print(f'req[{k}]={req[k]}')
                d, i = self.nntree.query(req[k])
                # print(f'd, i = {d}, {i}, self.indices[{i}]={self.indices[i]}')
                vals[k] = self.grid[tuple(self.indices[i])]
                # print(f'subgrid={self.grid[tuple(self.indices[i])]}')
                # print(f'vals[{k}]={vals[k]}')
        elif extrapolation_method == 'linear':
            for k in nan_indices:
                v = req[k]
                ic = np.array([np.searchsorted(self.axes[i], v[i])-1 for i in range(len(self.axes))])
                # print(f'v={v}, ic={ic}')

                # get the inferior corners of all nearest fully defined hypercubes; this
                # is all integer math so we can compare with == instead of np.isclose().
                sep = (np.abs(self.ics-ic)).sum(axis=1)
                corners = np.argwhere(sep == sep.min()).flatten()
                # print(f'corners={corners}')

                exvals = []
                for corner in corners:
                    slc = tuple([slice(self.ics[corner][i], self.ics[corner][i]+2) for i in range(len(self.ics[corner]))])
                    coords = [self.axes[i][slc[i]] for i in range(len(self.axes))]

                    # extrapolate:
                    lo = [c[0] for c in coords]
                    hi = [c[1] for c in coords]
                    subgrid = self.grid[slc]
                    pivot_indices = np.roll(np.arange(len(subaxes), -1, -1), -1)
                    fv = subgrid.transpose(*pivot_indices).reshape((2**req.shape[1], self.grid.shape[-1]))
                    # print(f'lo={lo}, hi={hi}')
                    # print(f'subgrid={subgrid}')
                    # print(f'fv={fv}')
                    exval = self.ndpolate(v, lo, hi, fv)
                    exvals.append(exval)
                    # print(f'exval={exval}\nexvals={exvals}')
                vals[k] = np.array(exvals).mean(axis=0)
                # print(f'np.array(exvals)={np.array(exvals)}, mean={np.array(exvals).mean(axis=0)}')
        else:
            raise ValueError(r'extrapolation_method={extrapolation_method} is not supported.')

        return (vals, nanmask) if return_nanmask else vals

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


def map_to_cube(v, axes, intervals, return_naxes=False):
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
    * `v` (2-D array): an array of vectors of interest in old coordinates
    * `axes` (tuple of arrays): a list of axes in old coordinates
    * `intervals` (tuple of 2-D arrays): an array of step sizes at the
      beginning and the end of the new axes.
    * `return_naxes` (bool, optional, default=False): whether a tuple of
      normalized axes should be returned as well.

    Returns
    -------
    * (array) a transformed 2-D array in new coordinates, or
    * (tuple) a transformed 2-D array in new coordinates and transformed axes,
      if `return_naxes=True`.
    """

    nv = np.empty_like(v)

    for k in range(v.shape[1]):
        ranges = (axes[k][0], axes[k][-1])
        delta_k = intervals[k][0] + (v[:,k]-ranges[0])/(ranges[1]-ranges[0])*(intervals[k][1]-intervals[k][0])
        nv[:,k] = (v[:,k]-ranges[0])/delta_k

    if return_naxes:
        naxes = [(axes[k]-axes[k][0]) / (intervals[k][0]+(axes[k]-axes[k][0])/(axes[k][-1]-axes[k][0])*(intervals[k][1]-intervals[k][0])) for k in range(len(axes))]
        return (nv, tuple(naxes))

    return nv


def kdtree(axes, grid, index_non_nans=True):
    """
    Construct a k-D tree for nearest neighbor lookup.

    k-dimensional trees are space-partitioning data structures for organizing
    points in a k-dimensional parameter space. They are very efficient for
    looking up nearest neighbors. This function takes a list of axes and the
    grid spun by the axes and it constructs a corresponding k-D tree.

    Grid is expected to be sparse, i.e.~not all vertices are expected to be
    defined. Undefined vertices in `grid` are flagged with `np.nan`.

    Parameters
    ----------

    * `axes` (tuple of arrays):
        a list of axes that span the grid
    * `grid` (ndarray):
        an N-dimensional grid of function values
    * `index_non_nans` (bool, optional, default=True)
        should non-nan values be indexed and returned to the calling function.
        If set to False, then only the kdtree is returned. If set to True,
        then an array of indices of non-nan elements is also returned.

    Returns
    -------
    * <scipy.spatial.cKDTree> instance if `index_non_nans`=False, or tuple
        (<scipy.spatial.cKDTree>, non_nan_indices) if `index_non_nans`=True.
    """

    non_nan_indices = np.argwhere(~np.isnan(grid))
    non_nan_vertices = np.array(
        [[axes[i][non_nan_indices[k][i]] for i in range(len(axes))] for k in range(len(non_nan_indices))])
    if index_non_nans:
        return cKDTree(non_nan_vertices, copy_data=True), non_nan_indices
    else:
        return cKDTree(non_nan_vertices, copy_data=True)


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


def find_nearest_hypercubes(nv, naxes, ics):
    ic = np.array([np.searchsorted(naxes[k], nv[:,k])-1 for k in range(len(naxes))]).T
    # print(f'ic.shape={ic.shape}')  # (5, 3)
    # print(f'ics.shape={ics.shape}')  # (2947, 3)
    seps = (np.abs(ics[:,None,:]-ic)).sum(axis=2)  # (2947, 5, 3) -> (2947, 5)
    corners = np.argwhere(seps == seps.min(axis=0)) # (N, 2)

    return corners