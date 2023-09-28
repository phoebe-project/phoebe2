#!/usr/bin/env python3

"""
interferometry.py

Interferometric module for Phoebe.

Reference: Hanbury-Brown et al. (1974, MNRAS 167, 121-136)

"""

import sys
import numpy as np
from scipy.special import j1
from astropy import units
from astropy import constants as const


def j32(x):
    """
    Bessel function of the first kind, half-order 3/2.
 
    Derived from:
                     2
    J_1/2(x) = sqrt ---- sin x,
                    pi x
   
    using recurrence relations:
   
                              2s
    J_{s-1}(x) + J_{s+1}(x) = -- J_s(x),
                              x
    and:
              1
    J_s'(x) = - J_{s-1}(x) - J_{s+1}(x).
              2
   
    Consequently:
                  s
    J_{s+-1}(x) = - J_s(x) -+ J_s'(x),
                  2
    and:
                    2     sin x
    J_{3/2} = sqrt ---- { ----- - cos x }.
                  pi x     x
    """

    eps = 1.0e-16

    if x > eps:
        return np.sqrt(2.0/(np.pi*x)) * (np.sin(x)/x - np.cos(x))
    else:
        return 0.0


def planck(T, lambda_):
    """
    Planck function, i.e., black-body intensity in J s^-1 sr^-1 m^-2 m^-1 units.

    """
    h = const.h.value
    c = const.c.value
    k_B = const.k_B.value
    return 2.0*h*c**2/lambda_**5 / (np.exp(h*c/(lambda_*k_B*T))-1.0)


def vis_simple(b, system, ucoord=None, vcoord=None, wavelengths=None, info={}):
    """
    Compute interferometric squared visibility |V|^2.
    A simple model of limb-darkened disk(s).

    b           .. Bundle object
    system      .. System object
    ucoord      .. baselines [m]
    vcoord      .. baselines [m]
    wavelengths .. wavelengths [m]
    info        .. dictionary w. 'original_index' to get (u, v)'s

    """

#    print("vis_simple")
#    print("b = ", b)
#    print("b['distance@system'] = ", b['distance@system'])
#    print("b['hierarchy@system'] = ", b['hierarchy@system'])
#    print("b['requiv@primary@star@component'] = ", b['requiv@primary@star@component'])
#    print("b['teff@primary@star@component'] = ", b['teff@primary@star@component'])
#    print("system = ", system)    
#    print("vars(system) = ", vars(system))    
#    print("system.bodies = ", system.bodies)
#    print("system.bodies[0] = ", system.bodies[0])
#    print("vars(system.bodies[0]) = ", vars(system.bodies[0]))
#    print("system.bodies[0]._mesh = ", system.bodies[0]._mesh)
#    print("system.xi = ", system.xi)    
#    print("info = ", info)
#    sys.exit(1)

#    val = 0.5; return {'vises': val}  # dbg

    # Note: b.get_value() call is extremely slow!!! cf. backend.py
    j = info['original_index']
    d = system.distance					# m
    u = ucoord[j]					# m
    v = vcoord[j]					# m
    lambda_ = wavelengths[j]				# m

    d *= (units.m/units.solRad).to('1')			# solRad
    B = np.sqrt(u**2 + v**2)				# m
    u /= lambda_					# cycles per baseline
    v /= lambda_					# cycles per baseline

    Lumtot = 0.0
    mutot = 0.0+0.0j

    for i, body in enumerate(system.bodies):

        phi = 2.0*body.requiv/d		# rad
        x = system.xi[i]/d		# rad
        y = system.yi[i]/d		# rad
        arg = np.pi*phi*B/lambda_	# rad

	# Note: limb-darkening should be passband (monochromatic)!
        coeff = body.ld_coeffs['bol'][0]
        alpha = 1.0-coeff
        beta = coeff

        # Note: luminosity should be passband & integrated over surface!
        Lum_lambda = np.pi*(body.requiv*units.solRad.to('m'))**2 * planck(lambda_, body.teff)

        mu = Lum_lambda * 1.0/(alpha/2.0 + beta/3.0)
        mu *= (alpha*j1(arg)/arg + beta*np.sqrt(np.pi/2.0)*j32(arg)/arg**(3.0/2.0))
        mu *= np.exp(-2.0*np.pi*(0.0+1.0j) * (u*x + v*y))

        mutot += mu
        Lumtot += Lum_lambda

    val = (abs(mutot)/Lumtot)**2

    return {'vises': val}


def vis_integrate(b, system, ucoord=None, vcoord=None, wavelengths=None, info={}):
    """
    Compute interferometric squared visibility |V|^2.
    A complex model w. integration over meshes.

    Note: see vis().

    """

    meshes = system.meshes
    components = info['component']
    dataset = info['dataset']

    visibilities = meshes.get_column_flat('visibilities', components)

    if np.all(visibilities==0):
        return {'vises': np.nan}

    abs_intensities = meshes.get_column_flat('abs_intensities:{}'.format(dataset), components)
    mus = meshes.get_column_flat('mus', components)
    areas = meshes.get_column_flat('areas_si', components)

    j = info['original_index']
    d = system.distance			# m
    u = ucoord[j]			# m
    v = vcoord[j]			# m
    lambda_ = wavelengths[j]		# m

    d *= (units.m/units.solRad).to('1')				# solRad
    centers = meshes.get_column_flat('centers', components)	# solRad
    xs = centers[:,0]						# solRad
    ys = centers[:,1]						# solRad
    x = xs/d							# rad
    y = ys/d							# rad
    u /= lambda_						# cycles per baseline
    v /= lambda_						# cycles per baseline

    Lum = abs_intensities*areas*mus*visibilities
    mu = Lum*np.exp(-2.0*np.pi*(0.0+1.0j) * (u*x + v*y))

    mutot = np.sum(mu)
    Lumtot = np.sum(Lum)

    val = (abs(mutot)/Lumtot)**2

    return {'vises': val}


vis = vis_integrate

