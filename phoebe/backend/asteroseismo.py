"""
This module exists because the definition of spherical harmonics used
in scipy and the ones commonly used in asteroseismology are slightly
different. Here we implement spherical harmonics after Arfken (1970),
summarized by Townsend (2003), MNRAS 343, 125.
"""

import numpy as np
from scipy.special import legendre
try:
    from scipy.misc import factorial
except:
    from scipy.special import factorial

def as_legendre(l, m, x):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @x: argument, typically cos \theta

    Computes associated Legendre polynomials P_l^m in x.

    This function uses scipy's version of Legendre polynomials while
    explicitly imposing the following relation for negative values of m:

      P_l^{-m}(x) = (-1)^m (l-m)!/(l+m)! P_l^m (x)
    """

    m_ = abs(m)
    legendre_poly = legendre(l)
    deriv_legpoly_ = legendre_poly.deriv(m=m_)
    deriv_legpoly = np.polyval(deriv_legpoly_,x)
    P_l_m = (-1)**m_ * (1-x**2)**(m_/2.) * deriv_legpoly
    if m < 0:
        P_l_m = (-1)**m_ * factorial(l-m_)/factorial(l+m_) * P_l_m
    return P_l_m

def as_Y(l, m, theta, phi):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @theta: colatitude, 0 at the +z pole, pi at the -z pole
    @phi: azimuth, 0 in the x direction, goes from 0 to 2pi

    Spherical harmonic according to Townsend (2003), MNRAS 343, 125:

      Y_l^m (\theta, \phi) = (-1)^m \sqrt{ (2l+1)/4pi (l-m)!/(l+m)! } P_l^m(cos\theta) exp(im\phi)
    """

    factor = (-1)**m * np.sqrt( (2*l+1)/(4*np.pi) * factorial(l-m)/factorial(l+m))
    Plm = as_legendre(l, m, np.cos(theta))
    return factor*Plm*np.exp(1j*m*phi)

def as_norm_J(l, m):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)

    Normalization factor for the derivatives.
    """

    return np.sqrt(float((l**2-m**2))/(4*l**2-1)) if abs(m) < l else 0

def as_norm_atlp1(l, m, Omega, k):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @Omega: spin parameter (Omega_rot/omega_freq)
    @k:

    Normalization factor for the l+1 term.
    """

    if Omega < 1e-6:
        return 0.0
    return Omega * (l-abs(m)+1)/(l+1) * 2./(2*l+1) * (1-l*k)

def as_norm_atlm1(l, m, Omega, k):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @Omega: spin parameter (Omega_rot/omega_freq)
    @k:

    Normalization factor for the l-1 term.
    """

    if Omega < 1e-6:
        return 0.0
    return Omega * (l+abs(m))/l * 2./(2*l+1) * (1 + (l+1)*k)

def as_dYdtheta(l, m, theta, phi):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @theta: colatitude, 0 at the +z pole, pi at the -z pole
    @phi: azimuth, 0 in the x direction, goes from 0 to 2pi

    The derivative of a spherical harmonic wrt colatitude.

      sin(theta)*dY/dtheta = (l*J_{l+1}^m * Y_{l+1}^m - (l+1)*J_l^m * Y_{l-1,m})
    """

    if abs(m) >= l:
        Y = 0.
    else:
        factor = 1./np.sin(theta)
        term1 = l     * as_norm_J(l+1, m) * as_Y(l+1, m, theta, phi)
        term2 = (l+1) * as_norm_J(l,   m) * as_Y(l-1, m, theta, phi)
        Y = factor * (term1 - term2)
    return Y

def as_dYdphi(l, m, theta, phi):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @theta: colatitude, 0 at the +z pole, pi at the -z pole
    @phi: azimuth, 0 in the x direction, goes from 0 to 2pi

    The derivative of a spherical harmonic wrt longitude.

      dY/dphi = i*m*Y
    """

    return 1j*m*as_Y(l, m, theta, phi)

def as_xi_r(l, m, theta, phi, wt):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @theta: colatitude, 0 at the +z pole, pi at the -z pole
    @phi: azimuth, 0 in the x direction, goes from 0 to 2pi
    @wt: omega*t, where omega=2 \pi \nu is the angular frequency, and t is time

    Unscaled radial displacement, see
        Aerts et al. (2010), the Asteroseismology book and
        Zima W, A&A 455, 227–234 (2006), Appendix A
    """

    return as_Y(l, m, theta, phi) * np.exp(1j*wt)

def as_xi_theta(l, m, theta, phi, wt, Omega=0, k=0):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @theta: colatitude, 0 at the +z pole, pi at the -z pole
    @phi: azimuth, 0 in the x direction, goes from 0 to 2pi
    @wt: omega*t, where omega=2 \pi \nu is the angular frequency, and t is time
    @Omega: spin parameter
    @k:

    Unscaled co-latitudinal displacement, see
        Aerts et al. (2010), the Asteroseismology book and
        Zima W, A&A 455, 227–234 (2006), Appendix A
    """

    term1 = k * as_dYdtheta(l, m, theta, phi) * np.exp(1j*wt)
    term2 = as_norm_atlp1(l, m, Omega, k) / np.sin(theta) * as_dYdphi(l+1, m, theta, phi) * np.exp(1j*wt + np.pi/2)
    term3 = as_norm_atlm1(l, m, Omega, k) / np.sin(theta) * as_dYdphi(l-1, m, theta, phi) * np.exp(1j*wt - np.pi/2)
    return term1 + term2 + term3

def as_xi_phi(l, m, theta, phi, wt, Omega=0, k=0):
    """
    @l: non-radial degree (the number of longitudinal and latitudinal node lines)
    @m: longitudinal order (the number of longitudinal node lines)
    @theta: colatitude, 0 at the +z pole, pi at the -z pole
    @phi: azimuth, 0 in the x direction, goes from 0 to 2pi
    @wt: omega*t, where omega=2 \pi \nu is the angular frequency, and t is time
    @Omega: spin parameter
    @k:

    Unscaled longitudinal displacement, see
        Aerts et al. (2010), the Asteroseismology book and
        Zima W, A&A 455, 227–234 (2006), Appendix A
    """

    term1 = k/np.sin(theta) * as_dYdphi(l, m, theta, phi) * np.exp(1j*wt)
    term2 = -as_norm_atlp1(l, m, Omega, k) * as_dYdtheta(l+1, m, theta, phi)*np.exp(1j*wt+np.pi/2)
    term3 = -as_norm_atlm1(l, m, Omega, k) * as_dYdtheta(l-1, m, theta, phi)*np.exp(1j*wt-np.pi/2)
    return term1 + term2 + term3

if __name__=='__main__':
    print('Checking whether negative m values are computed correctly:'),
    ls, x = [0, 1, 2, 3, 4, 5], np.cos(np.linspace(0, np.pi, 100))
    check = 0
    for l in ls:
        for m in range(-l, l+1, 1):
            Ppos = as_legendre(l,  m, x)
            Pneg = as_legendre(l, -m, x)
            mycheck = Pneg, (-1)**m * factorial(l-m)/factorial(l+m) * Ppos
            check += sum(abs(mycheck[0]-mycheck[1])>1e-10)
    if check == 0:
        print('yes')
    else:
        print('no -- exiting')
        exit()
