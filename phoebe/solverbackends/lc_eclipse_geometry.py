import numpy as np
try:
    from scipy.signal import find_peaks
    from scipy.optimize import newton
except ImportError:
    _can_compute_eclipse_params = False
else:
    _can_compute_eclipse_params = True

def compute_eclipse_params(phase, flux):
    if not _can_compute_eclipse_params:
        raise ImportError("could not import scipy.signal.find_peaks and scipy.optimize.newton")

    # compute the primary gradient and differences between consecutive points in the gradient
    # mask only those larger than the mean (this filters out the eclipses)
    # the light curve has to be smooth, uniformly sampled and phased on range (0,1), with 0 corresponding to supconj
    # NOTE: ONLY WORKS WELL IF NO OR SMALL ELLIPSOIDAL VARIATIONS, fails if ELV > ECL
    grad1 = np.gradient(flux)/np.gradient(flux).max()
    diffs = np.array([grad1[i+1]-grad1[i] for i in range(len(grad1)-1)])
    mask = (np.abs(diffs) >= np.mean(np.abs(diffs)))
    # to isolate individual eclipses, compute diffs in phase to identify groups and breaks
    phmask = np.hstack((phase[:-1][mask],phase[-1]))
    phdiffs = [phmask[i+1]-phmask[i] for i in range(len(phmask)-1)]
    breaks = np.argwhere(phdiffs>np.mean(phdiffs)).flatten()
    # eclipses
    # primary eclipse
    phases1 = np.hstack((phase[1:][mask][breaks[1]+1:] - 1., phase[0],phase[1:][mask][:breaks[0]+1]))
    fluxes1 = np.hstack((flux[1:][mask][breaks[1]+1:], flux[0], flux[1:][mask][:breaks[0]+1]))
    phases2 = phase[1:][mask][breaks[0]+1:breaks[1]+1]
    fluxes2 = flux[1:][mask][breaks[0]+1:breaks[1]+1]
    # positions
    pos_primary = 0.5*(phases1.max() + phases1.min())
    pos_secondary = 0.5*(phases2.max() + phases2.min())
    # widths
    primary_width = (phases1.max() - phases1.min())
    secondary_width = (phases2.max() - phases2.min())
    # depths
    # very simple implementation using the datapoints
    # could be extended to more advanced with quadratic fitting or interpolation
    depth_primary = flux[(np.abs(phase - pos_primary)).argmin()]
    depth_secondary = flux[(np.abs(phase - pos_secondary)).argmin()]
    # if diagnose:
    #     plt.plot(phase[phase<=0.5], flux[phase<=0.5], 'k.')
    #     plt.plot(phase[phase>=0.5]-1, flux[phase>=0.5], 'k.')
    #     plt.plot(phases1, fluxes1, '.', label='primary')
    #     plt.plot(phases2[phases2<=0.5], fluxes2[phases2<=0.5], '.', c='orange', label='secondary')
    #     plt.plot(phases2[phases2>=0.5]-1, fluxes2[phases2>=0.5], '.', c='orange')
    #     plt.axvline(x=pos_primary, c='blue', ls='--', label='primary pos')
    #     plt.axvline(x=pos_secondary, c='orange', ls='--', label='secondary pos')
    #     plt.legend()
    #     plt.show()
    return {
        'primary_width': primary_width,
        'secondary_width': secondary_width,
        'primary_position': pos_primary,
        'secondary_position': pos_secondary,
        'primary_depth': depth_primary,
        'secondary_depth': depth_secondary
    }


def ecc_w_from_geometry(dphi):


    "dphi =  secondary_position - primary_position (returned from  compute_eclipse_params)"

    def compute_psi(psi, deltaPhi):
        return psi - np.sin(psi) - 2*np.pi*deltaPhi

    def ecc_func(psi):
        return np.sin(0.5*(psi-np.pi))*(1.-0.5*(np.cos(0.5*(psi-np.pi)))**2)**(-0.5)

    def argper(ecc, psi):
        if ecc <= 0.:
            return 0.
        return np.arccos(1./ecc * (1.-ecc**2)**0.5 * np.tan(0.5*(psi-np.pi)))

    psi = newton(compute_psi, 0.5, args=(dphi,))
    ecc = np.abs(ecc_func(psi))
    w = argper(ecc, psi)
    return ecc, w
