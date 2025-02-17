import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger("CONTACT ENVELOPES")
logger.addHandler(logging.NullHandler())


def _isolate_neck(coords_all, teffs_all, cutoff=0., component=1, plot=False):
    """
    Selects vertices in the x = [0 + cutoff, 1 - cutoff] region of the Roche geometry.

    Parameters
    ----------
    coords_all: 3D points
    teffs_all: Teffs to filter on
    cutoff: wiggle room value for the edges (default=0)
    component: which component? 1 or 2 (default=1)
    plot: plot the resulting selection? (default=False)

    Returns
    -------
    Filtered coordinates, filtered Teffs, indices of coords_all that was filtered on
    """
    if component == 1:
        cond = coords_all[:, 0] >= 0 + cutoff
    elif component == 2:
        cond = coords_all[:, 0] <= 1 - cutoff
    else:
        raise ValueError

    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axes[0].scatter(coords_all[cond][:, 0], coords_all[cond][:, 1])
        axes[1].scatter(coords_all[cond][:, 0], teffs_all[cond])
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('teff')
        fig.tight_layout()
        plt.show()

    return coords_all[cond], teffs_all[cond], np.argwhere(cond).flatten()


def _dist(p1, p2):
    """
    Euclidean distance between two points in R^n
    Parameters
    ----------
    p1: point 1
    p2: point 2

    Returns
    -------
    Square root of dot product of p2 - p1
    """
    return np.sqrt(np.dot(p2 - p1, p2 - p1))


def _isolate_sigma_fitting_regions(coords_neck, teffs_neck, direction='x', cutoff=0., component=1, plot=False):
    """
    Determine the region on which sigma smoothing is going to be fitted. If `direction=="y"`, this is going to be the
    slice close to `x=0` (or `x=1` for the secondary). If `direction=="x"`, this is going to be the slice near the
    equator (`y=0`).

    Parameters
    ----------
    coords_neck: 3D points of the neck region
    teffs_neck: Teffs of the neck region
    direction: which direction? either 'x' or 'y'
    cutoff: wiggle room value for the edges (default=0)
    component: which component? 1 or 2 (default=1)
    plot: plot the resulting selection? (default=False)

    Returns
    -------
    Filtered coordinates, filtered Teffs
    """
    distances = [_dist(p1, p2) for p1, p2 in combinations(coords_neck, 2)]
    min_dist = np.min(distances[distances != 0])  # smallest dist. between any two vertices (~ resolution of the mesh)

    if direction == 'x':
        cond = (coords_neck[:, 1] >= -cutoff - 0.2 * min_dist) & (coords_neck[:, 1] <= cutoff + 0.2 * min_dist)

    elif direction == 'y':
        if component == 1:
            cond = coords_neck[:, 0] <= 0 + cutoff + 0.15 * min_dist
        elif component == 2:
            cond = coords_neck[:, 0] >= 1 - cutoff - 0.15 * min_dist
        else:
            raise ValueError('Unknown component, must be 1 or 2')
    else:
        raise ValueError('Unknown direction, must be either "x" or "y"')

    if plot:
        if direction == 'x':
            plt.scatter(coords_neck[cond][:, 0], teffs_neck[cond])
            plt.show()
        elif direction == 'y':
            plt.scatter(coords_neck[cond][:, 1], teffs_neck[cond])
            plt.show()

    return coords_neck[cond], teffs_neck[cond]


def _compute_new_teff_at_neck(coords1, teffs1, coords2, teffs2, w=0.5):
    """
    Computes the target teff at the neck that should be attained by the smoothing.
    Parameters
    ----------
    coords1: Roche coordinates of the primary
    teffs1: Teffs of the primary
    coords2: Roche coordinates of the secondary
    teffs2: Teffs of the secondary
    w: Weight of the primary contribution relative to the secondary

    Returns
    -------
    x coordinate of the neck, target Teff at the neck
    """
    distances1 = [_dist(p1, p2) for p1, p2 in combinations(coords1, 2)]
    min_dist1 = np.min(distances1[distances1 != 0])
    distances2 = [_dist(p1, p2) for p1, p2 in combinations(coords2, 2)]
    min_dist2 = np.min(distances2[distances2 != 0])

    x_neck = np.average((coords1[:, 0].max(), coords2[:, 0].min()))

    teffs_neck1 = teffs1[coords1[:, 0] >= coords1[:, 0].max() - 0.25 * min_dist1]
    teffs_neck2 = teffs2[coords2[:, 0] <= coords2[:, 0].min() + 0.25 * min_dist2]

    teff1 = np.max(teffs2)  # np.average(teffs_neck1)
    teff2 = np.average(teffs_neck2)
    tavg = w * teff1 + (1 - w) * teff2
    if tavg > teffs2.max():
        logger.warning('Tavg > Teff2, setting new temperature to 1 percent of Teff2 max. %i > %i' % (
            int(tavg), int(teffs2.max())))
        tavg = teffs2.max() - 0.01 * teffs2.max()

    return x_neck, tavg


def _compute_sigmax(Tavg, x, x0, offset, amplitude):
    """
    Rather than fitting for sigma_x, we simply invert a gaussian profile to match the target Tavg
    (minus a potential offset)
    """
    return ((-1) * (x - x0) ** 2 / np.log((Tavg - offset) / amplitude)) ** 0.5


def _fit_sigma(coords, teffs, offset=0., cutoff=0., direction='y', component=1, plot=False):
    """
    Fits the width of a gaussian distribution to given coords and teffs.

    Parameters
    ----------
    coords: 3D coordinates in the Roche geometry
    teffs: teffs at the given coordinates
    offset: temperature offset to include (default=0)
    cutoff: x-displacement of the center of the gaussian (default=0)
    direction: direction to fit gaussian in (default='y')
    component: which component? 1 or 2 (default=1)
    plot: plot the resulting computation? (default=False)

    Returns
    -------
    the computed width, amplitude, and the fitted model Teffs
    """

    def gaussian_1d(x, s):
        a = 1. / s ** 2
        g = offset + amplitude * np.exp(- (a * ((x - x0) ** 2)))
        return g

    if direction == 'y':
        coord_ind = 1
        x0 = 0.
    elif direction == 'x':
        coord_ind = 0
        if component == 1:
            x0 = 0 + cutoff
        elif component == 2:
            x0 = 1 - cutoff
        else:
            raise ValueError('Unknown component selected, must be "1" or "2"')
    else:
        raise ValueError('Unknown direction selected, must be "y" or "x"')

    amplitude = teffs.max() - offset
    sigma_0 = 0.5
    result = curve_fit(gaussian_1d, xdata=coords[:, coord_ind], ydata=teffs, p0=(sigma_0,), bounds=[0.01, 1000])

    sigma = result[0]
    model = gaussian_1d(coords[:, coord_ind], sigma)

    if plot:
        plt.scatter(coords[:, coord_ind], teffs)
        plt.scatter(coords[:, coord_ind], model)
        plt.show()

    return sigma, amplitude, model


def _twoD_Gaussian(coords, sigma_x, sigma_y, amplitude, cutoff=0., offset=0., component=1):
    """
    Simple 2D gaussian profile with widths sigma_x and sigma_y and an amplitude.
    If amplitude is the maximal temperature of a contact component, and the sigmas the width their respective calculated
    gradients, this function returns the smoothed temperatures of the component.

    Parameters
    ----------
    coords: 3D coordinates in roche geometry
    sigma_x: width of gaussian in x-direction
    sigma_y: width of gaussian in y-direction
    amplitude: amplitude of gaussian
    cutoff: x-displacement of the center of the gaussian (default=0)
    offset: constant added to the gaussian (default=0)
    component: which component of the binary (default=1)

    Returns
    -------
    2D gaussian profile
    """
    y0 = 0.
    x0 = 0 + cutoff if component == 1 else 1 - cutoff
    a = 1. / sigma_x ** 2
    b = 1. / sigma_y ** 2
    return offset + amplitude * np.exp(- (a * ((coords[:, 0] - x0) ** 2))) * np.exp(- (b * ((coords[:, 1] - y0) ** 2)))


def gaussian_smoothing(xyz1, teffs1, xyz2, teffs2, w=0.5, cutoff=0., offset=0.):
    """
    Adapts the temperatures of the primary and the secondary to blend nicely using gaussian smoothing.
    It calculates an average temperature at the neck, appropriate widths of the gaussian profiles (essentially the temp-
    gradient), and applies the smoothing.

    Parameters
    ----------
    xyz1: 3D coordinates in roche geometry of the primary
    teffs1: teffs of the primary
    xyz2: 3D coordinates in roche geometry of the secondary
    teffs2: teffs of the secondary
    w: contribution of the primary teff vs the secondary (a weighting factor, default=0.5)
    cutoff: x-displacement of the center of the gaussians toward the center of mass (default=0)
    offset: constant added to the gaussian (default=0)

    Returns
    -------

    """
    coords_neck_1, teffs_neck_1, cond1 = _isolate_neck(xyz1, teffs1, cutoff=cutoff, component=1, plot=False)
    coords_neck_2, teffs_neck_2, cond2 = _isolate_neck(xyz2, teffs2, cutoff=cutoff, component=2, plot=False)

    x_neck, Tavg = _compute_new_teff_at_neck(coords_neck_1, teffs_neck_1, coords_neck_2, teffs_neck_2, w=w,
                                             offset=offset)

    sigma_x1 = _compute_sigmax(Tavg, x_neck, x0=0 + cutoff, offset=offset, amplitude=teffs_neck_1.max())
    sigma_x2 = _compute_sigmax(Tavg, x_neck, x0=1 - cutoff, offset=offset, amplitude=teffs_neck_2.max())

    coords_fit_y1, teffs_fit_y1 = _isolate_sigma_fitting_regions(coords_neck_1, teffs_neck_1, direction='y',
                                                                 cutoff=cutoff, component=1, plot=False)
    coords_fit_y2, teffs_fit_y2 = _isolate_sigma_fitting_regions(coords_neck_2, teffs_neck_2, direction='y',
                                                                 cutoff=cutoff, component=2, plot=False)

    sigma_y1, amplitude_y1, model_y1 = _fit_sigma(coords_fit_y1, teffs_fit_y1, offset, direction='y',
                                                  component=1, plot=False)
    sigma_y2, amplitude_y2, model_y2 = _fit_sigma(coords_fit_y2, teffs_fit_y2, offset, direction='y',
                                                  component=2, plot=False)

    new_teffs1 = _compute_twoD_Gaussian(coords_neck_1, sigma_x1, sigma_y1, teffs_neck_1.max(), cutoff=cutoff,
                                        offset=offset, component=1)
    new_teffs2 = _compute_twoD_Gaussian(coords_neck_2, sigma_x2, sigma_y2, teffs_neck_2.max(), cutoff=cutoff,
                                        offset=offset, component=2)

    # print(cond1, len(cond1), len(new_teffs1))
    # print(cond2, len(cond2), len(new_teffs2))
    teffs1[cond1] = new_teffs1
    teffs2[cond2] = new_teffs2

    return teffs1, teffs2


def lateral_transfer(t2s, teffs2, mixing_power, teff_ratio):
    """
    Scales the temperatures of the secondary to that of the primary only in a horizontal band the size of the contact's
    neck. This implies mixing occurs due to mass transfer across the neck.
    """
    x2s = t2s[:, 0]
    y2s = t2s[:, 1]
    z2s = t2s[:, 2]

    y2s_neck = y2s[x2s < 1]
    z2s_neck = z2s[x2s < 1]
    rs_neck = (y2s_neck ** 2 + z2s_neck ** 2) ** 0.5
    lat = np.min(rs_neck)
    assert lat == np.min(z2s_neck)
    filt = (z2s > -lat) & (z2s < lat)  # select band extending the (projected) height of the neck
    c = (lat - np.abs(z2s[filt])) ** mixing_power
    latitude_dependence = c / c.max()
    teffs2[filt] *= 1 + (1 - teff_ratio) * latitude_dependence

    return teffs2


def isotropic_transfer(t2s, teffs2, mixing_power, teff_ratio):
    """
    Scales the temperatures of the secondary to that of the primary, parametrized by the radial distance from the
    origin (which is the center of the primary). Implies mixing occurs diffusively from th center of the neck.
    """
    d2s = np.sqrt(t2s[:, 0] * t2s[:, 0] + t2s[:, 1] * t2s[:, 1] + t2s[:, 2] * t2s[:, 2])
    teffs2 *= 1 + (1 - teff_ratio) * (1 - ((d2s - d2s.min()) / (d2s.max() - d2s.min()))) ** mixing_power
    return teffs2


def perfect_transfer(t2s, teff2s, teff_ratio):
    """
    Scales the temperatures of the secondary to that of the primary, implying perfect thermal mixing occurred deep in
    the interior of the stars, and little surface mixing occurs.
    """
    teff2s *= 1 / teff_ratio
    return teff2s


def mix_teffs(xyz1, teffs1, xyz2, teffs2, mixing_method='lateral', mixing_power=0.5, teff_ratio=1.):
    """
    Applies a temperature mixing, primarily of component 2, according to some mixing method and other parameters.
    If `mixing_method == 'smoothing'`, simple gaussian smoothing is applied rather than an energy transfer model

    Parameters
    ----------
    xyz1: 3D roche coordinates of the primary
    teffs1: Teffs of the primary (these are only altered by smoothing)
    xyz2: 3D roche coordinates of the secondary
    teffs2: Teffs of the secondary (these are altered by all mixing methods)
    mixing_method: model of energy transfer, 'lateral', 'isotropic', 'spotty', 'perfect' or 'smoothing'
        (default='lateral')
    mixing_power: parameter value of the mixing efficiency of the lateral and isotropic mixing model.
    teff_ratio: ratio of (averaged/global) secondary Teff over (averaged/global) primary Teff (pre-mixing of course).
        This value determines temperature gradient which drives the mixing. (E.g. if `teff_ratio==1` then no mixing is
        to be applied, since there's no expected energy transfer)

    Returns
    -------
    modified Teffs of the primary, modified Teffs of the secondary
    """
    if mixing_method == 'lateral':
        teffs2 = lateral_transfer(xyz2, teffs2, mixing_power, teff_ratio)
    elif mixing_method == 'isotropic':
        teffs2 = isotropic_transfer(xyz2, teffs2, mixing_power, teff_ratio)
    elif mixing_method == 'perfect':
        teffs2 = perfect_transfer(xyz2, teffs2, teff_ratio)
    elif mixing_method == 'smoothing':
        teffs1, teffs2 = gaussian_smoothing(xyz1, teffs1, xyz2, teffs2)
    else:
        raise ValueError('`mixing_method` must be "lateral", "isotropic", "perfect" or "smoothing"')
    return teffs1, teffs2
