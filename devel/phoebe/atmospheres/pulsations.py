"""
Calculate the stellar surface displacements due to pulsations.

Main references: Aerts 1993, Zima 2006, Townsend 2003.

The surface displacements and perturbations of observables are computable in the
nonrotating case, slow rotation with first order Coriolis effects taken into
account (and slow rotation in the traditional approximation, as these are
just a linear combination of the nonrotating case).
"""
import numpy as np
from numpy import sqrt,pi,sin,cos,exp
from scipy.special import lpmv,legendre
from scipy.special import sph_harm as sph_harm0
from scipy.misc.common import factorial
from scipy.integrate import dblquad,quad
from scipy.spatial import Delaunay
from phoebe.utils import Ylm

#{ Helper functions

def legendre_(l,m,x):
    """
    Legendre polynomial.
    
    Check equation (3) from Townsend, 2002:
    
    >>> ls,x = [0,1,2,3,4,5],cos(linspace(0,pi,100))
    >>> check = 0
    >>> for l in ls:
    ...     for m in range(-l,l+1,1):
    ...         Ppos = legendre_(l,m,x)
    ...         Pneg = legendre_(l,-m,x)
    ...         mycheck = Pneg,(-1)**m * factorial(l-m)/factorial(l+m) * Ppos
    ...         check += sum(abs(mycheck[0]-mycheck[1])>1e-10)
    >>> print check
    0
    """
    m_ = abs(m)
    legendre_poly = legendre(l)
    deriv_legpoly_ = legendre_poly.deriv(m=m_)
    deriv_legpoly = np.polyval(deriv_legpoly_,x)
    P_l_m = (-1)**m_ * (1.-x**2)**(m_/2.) * deriv_legpoly
    if m<0:
        P_l_m = (-1)**m_ * float(factorial(l-m_)/factorial(l+m_)) * P_l_m
    return P_l_m

def sph_harm(theta,phi,l=2,m=1):
    """
    Spherical harmonic according to Townsend, 2002.
    
    This function is memoized: once a spherical harmonic is computed, the
    result is stored in memory
    
    >>> theta,phi = mgrid[0:pi:20j,0:2*pi:40j]
    >>> Ylm20 = sph_harm(theta,phi,2,0)
    >>> Ylm21 = sph_harm(theta,phi,2,1)
    >>> Ylm22 = sph_harm(theta,phi,2,2)
    >>> Ylm2_2 = sph_harm(theta,phi,2,-2)
    >>> p = figure()
    >>> p = gcf().canvas.set_window_title('test of function <sph_harm>')
    >>> p = subplot(411);p = title('l=2,m=0');p = imshow(Ylm20.real,cmap=cm.RdBu)
    >>> p = subplot(412);p = title('l=2,m=1');p = imshow(Ylm21.real,cmap=cm.RdBu)
    >>> p = subplot(413);p = title('l=2,m=2');p = imshow(Ylm22.real,cmap=cm.RdBu)
    >>> p = subplot(414);p = title('l=2,m=-2');p = imshow(Ylm2_2.real,cmap=cm.RdBu)
    """
    #factor = (-1)**((m+abs(m))/2.) * sqrt( (2*l+1.)/(4*pi) * float(factorial(l-abs(m))/factorial(l+abs(m))))
    #Plm = legendre_(l,np.abs(m),cos(theta))
    #return factor*Plm*exp(1j*m*phi)
    #-- the above seems to be unstable for l~>30, but below is equivalent to:
    #if np.abs(m)>l:
    #    return np.zeros_like(phi)
    #return -sph_harm0(m,l,phi,theta)
    #-- due to vectorization, this seemed to be incredibly slow though. third
    #   try:
    if np.abs(m)>l:
        return np.zeros_like(phi)
    return -Ylm.Ylm(l,m,phi,theta)

def dsph_harm_dtheta(theta,phi,l=2,m=1):
    """
    Derivative of spherical harmonic wrt colatitude.
    
    Using Y_l^m(theta,phi).
    
    Equation::
        
        sin(theta)*dY/dtheta = (l*J_{l+1}^m * Y_{l+1}^m - (l+1)*J_l^m * Y_{l-1,m})
        
    E.g.: Phd thesis of Joris De Ridder
    """
    if abs(m)>=l:
        Y = 0.
    else:
        factor = 1./sin(theta)
        term1 = l     * norm_J(l+1,m) * sph_harm(theta,phi,l+1,m)
        term2 = (l+1) * norm_J(l,m)   * sph_harm(theta,phi,l-1,m)
        Y = factor * (term1 - term2)
    return Y/sin(theta)

def dsph_harm_dphi(theta,phi,l=2,m=1):
    """
    Derivative of spherical harmonic wrt longitude.
    
    Using Y_l^m(theta,phi).
    
    Equation::
        
        dY/dphi = i*m*Y
    """
    return 1j*m*sph_harm(theta,phi,l,m)
    

def norm_J(l,m):
    """
    Normalisation factor
    """
    if abs(m)<l:
        J = sqrt( (l**2.-m**2.)/(4*l**2-1.))
    else:
        J = 0.
    return J

def norm_atlp1(l,m,Omega,k):
    """
    Omega is actually spin parameter (Omega_rot/omega_freq)
    """
    return Omega * (l-abs(m)+1.)/(l+1.) * 2./(2*l+1.) * (1-l*k)

def norm_atlm1(l,m,Omega,k):
    """
    Omega is actually spin parameter (Omega_rot/omega_freq)
    """
    return Omega * (l+abs(m)+0.)/l * 2./(2*l+1.) * (1 + (l+1)*k)
    
#}

#{ Displacement fields

def radial(theta,phi,l,m,freq,t):
    """
    Radial displacement, see Zima 2006.
    
    t in period units (t=1/freq equals 2pi radians, end of one cycle)
    """
    return sph_harm(theta,phi,l,m) * exp(1j*2*pi*freq*t)

def colatitudinal(theta,phi,l,m,freq,t,Omega,k):
    """
    Colatitudinal displacement.
    
    Equation::
    
        ksi_theta = (asl      * k        * dY^m_l    /dtheta) * exp(2pi i t) 
            +   (a_{tl+1} /sin(theta)* dY^m_{l+1}/dphi)   * exp(2pi i t+pi/2) 
            +   (a_{tl-1} /sin(theta)* dY^m_{l-1}/dphi)   * exp(2pi i t-pi/2) 
            
    """
    term1 = k * dsph_harm_dtheta(theta,phi,l,m) * exp(1j*2*pi*freq*t)
    term2 = norm_atlp1(l,m,Omega,k) / sin(theta) * dsph_harm_dphi(theta,phi,l+1,m)  * exp(1j*2*pi*freq*t + pi/2)
    term3 = norm_atlm1(l,m,Omega,k) / sin(theta) * dsph_harm_dphi(theta,phi,l-1,m)  * exp(1j*2*pi*freq*t - pi/2)
    return term1 + term2 + term3

def longitudinal(theta,phi,l,m,freq,t,Omega,k):
    """
    Longitudinal displacement.
    
    ksi_phi = (asl   * k/sin(theta)* dY^m_l    /dphi)   * exp(2pi i t) 
          -   (a_{tl+1}            * dY^m_{l+1}/dtheta) * exp(2pi i t+pi/2) 
          -   (a_{tl-1}            * dY^m_{l-1}/dtheta) * exp(2pi i t-pi/2) 
    """
    term1 = k /sin(theta) * dsph_harm_dphi(theta,phi,l,m)*exp(1j*2*pi*freq*t)
    term2 = -norm_atlp1(l,m,Omega,k) * dsph_harm_dtheta(theta,phi,l+1,m)*exp(1j*2*pi*freq*t+pi/2)
    term3 = -norm_atlm1(l,m,Omega,k) * dsph_harm_dtheta(theta,phi,l-1,m)*exp(1j*2*pi*freq*t-pi/2)
    return term1 + term2 + term3

def surface(radius,theta,phi,t,l,m,freq,Omega,k,asl):
    ksi_r = 0.
    ksi_theta = 0.
    ksi_phi = 0.
    velo_r = 0.
    velo_theta = 0.
    velo_phi = 0.
    zero = np.zeros_like(theta)
    
    for il,im,ifreq,iOmega,ik,iasl in zip(l,m,freq,Omega,k,asl):
        #-- radial perturbation
        ksi_r_ = iasl*radius*sqrt(4*pi)*radial(theta,phi,il,im,ifreq,t)
        #-- add to the total perturbation of the radius and velocity
        ksi_r += ksi_r_
        velo_r += 1j*2*pi*ifreq*ksi_r_
        #-- colatitudinal and longitudonal perturbation when l>0
        norm = sqrt(4*pi)
        if il>0:
            ksi_theta_ = iasl*norm*colatitudinal(theta,phi,il,im,ifreq,t,iOmega,ik)
            ksi_phi_   = iasl*norm* longitudinal(theta,phi,il,im,ifreq,t,iOmega,ik)
            ksi_theta += ksi_theta_
            ksi_phi += ksi_phi_
            velo_theta += 1j*2*pi*ifreq*ksi_theta_
            velo_phi   += 1j*2*pi*ifreq*ksi_phi_
        else:
            ksi_theta += zero
            ksi_phi += zero
            velo_theta += zero
            velo_phi += zero
    
    return (radius+ksi_r.real),\
           (theta + ksi_theta.real),\
           (phi + ksi_phi.real),\
           velo_r.real,velo_theta.real,velo_phi.real

def observables(radius,theta,phi,teff,logg,t,l,m,freq,Omega,k,asl,delta_T,delta_g):
    """
    Good defaults:
    
    Omega = 0.1
    k = 1.0
    asl = 0.2
    radius = 1.
    delta_T=0.05+0j
    delta_g=0.0001+0.5j
    """
    gravity = 10**(logg-2)
    ksi_r = 0.
    ksi_theta = 0.
    ksi_phi = 0.
    velo_r = 0.
    velo_theta = 0.
    velo_phi = 0.
    ksi_grav = 0.
    ksi_teff = 0.
    zero = np.zeros_like(phi)
    
    for il,im,ifreq,iOmega,ik,iasl,idelta_T,idelta_g in \
       zip(l,m,freq,Omega,k,asl,delta_T,delta_g):
        rad_part = radial(theta,phi,il,im,ifreq,t)
        ksi_r_ = iasl*sqrt(4*pi)*rad_part#radial(theta,phi,il,im,ifreq,t)
        ksi_r += ksi_r_*radius
        velo_r += 1j*2*pi*ifreq*ksi_r_*radius
        if il>0:
            ksi_theta_ = iasl*sqrt(4*pi)*colatitudinal(theta,phi,il,im,ifreq,t,iOmega,ik)
            ksi_phi_ = iasl*sqrt(4*pi)*longitudinal(theta,phi,il,im,ifreq,t,iOmega,ik)
            ksi_theta += ksi_theta_
            ksi_phi += ksi_phi_
            velo_theta += 1j*2*pi*ifreq*ksi_theta_
            velo_phi += 1j*2*pi*ifreq*ksi_phi_
        else:
            ksi_theta += zero
            ksi_phi += zero
            velo_theta += zero
            velo_phi += zero
        ksi_grav += idelta_g*rad_part*gravity
        ksi_teff += idelta_T*rad_part*teff   
        
    return (radius+ksi_r.real),\
           (theta + ksi_theta.real),\
           (phi + ksi_phi.real),\
           velo_r.real,velo_theta.real,velo_phi.real,\
           (teff + ksi_teff.real),\
           np.log10(gravity+ksi_grav.real)+2
