#!/usr/bin/python
from numpy import *

def xfact(m):
    # computes (2m-1)!!/sqrt((2m)!)
    res = 1.
    for i in range(1,2*m+1):
        if i % 2: res *= i # (2m-1)!!
        res /= sqrt(i) # sqrt((2m)!)
    return res

def lplm_n(l,m,x):
    # associated legendre polynoms normalized as in Ylm
    l,m = int(l),int(m)
    assert 0<=m<=l and all(abs(x)<=1.)

    norm = sqrt(2*l+1) / sqrt(4*pi)
    if m == 0:
        pmm = norm * ones_like(x)
    else:
        pmm = (-1)**m * norm * xfact(m) * (1-x**2)**(m/2.)
    if l == m:
        return pmm
    pmmp1 = x*pmm*sqrt(2*m+1)
    if l == m+1:
        return pmmp1
    for ll in range(m+2,l+1):
        pll = (x*(2*ll-1)*pmmp1 - sqrt( (ll-1)**2 - m**2)*pmm)/sqrt(ll**2-m**2)
        pmm = pmmp1
        pmmp1 = pll
    return pll

def Ylm(l,m,phi,theta):
    # spherical harmonics
    # theta is from 0 to pi with pi/2 on equator
    l,m = int(l),int(m)
    assert 0 <= abs(m) <=l
    if m > 0:
        return lplm_n(l,m,cos(theta))*exp(1J*m*phi)
    elif m < 0:
        return (-1)**m*lplm_n(l,-m,cos(theta))*exp(1J*m*phi)
    return lplm_n(l,m,cos(theta))*ones_like(phi)

def Ylmr(l,m,phi,theta):
    # real spherical harmonics
    # theta is from 0 to pi with pi/2 on equator
    l,m = int(l),int(m)
    assert 0 <= abs(m) <=l
    if m > 0:
        return lplm_n(l,m,cos(theta))*cos(m*phi)*sqrt(2)
    elif m < 0:
        return (-1)**m*lplm_n(l,-m,cos(theta))*sin(-m*phi)*sqrt(2)
    return lplm_n(l,m,cos(theta))*ones_like(phi)

if __name__ == "__main__":
    from scipy.special import sph_harm
    from scipy.misc import factorial2, factorial
    from timeit import Timer
     
    def ref_xfact(m):
        return factorial2(2*m-1)/sqrt(factorial(2*m))
 
    print("Time: xfact(10)", Timer("xfact(10)",
        "from __main__ import xfact, ref_xfact").timeit(100))
    print("Time: ref_xfact(10)", Timer("ref_xfact(10)",
        "from __main__ import xfact, ref_xfact").timeit(100))
    print("Time: xfact(80)", Timer("xfact(80)",
        "from __main__ import xfact, ref_xfact").timeit(100))
    print("Time: ref_xfact(80)", Timer("ref_xfact(80)",
        "from __main__ import xfact, ref_xfact").timeit(100))
    
    print("m", "xfact", "ref_xfact") 
    for m in list(range(10)) + list(range(80,90)):
        a = xfact(m)
        b = ref_xfact(m)
        print(m, a, b)

    phi, theta = ogrid[0:2*pi:10j,-pi/2:pi/2:10j]

    print("Time: Ylm(1,1,phi,theta)", Timer("Ylm(1,1,phi,theta)",
        "from __main__ import Ylm, sph_harm, phi, theta").timeit(10))
    print("Time: sph_harm(1,1,phi,theta)", Timer("sph_harm(1,1,phi,theta)",
        "from __main__ import Ylm, sph_harm, phi, theta").timeit(10))
    
    print("l", "m", "max|Ylm-sph_harm|")
    for l in range(0,5):
        for m in range(-l,l+1):
            a = Ylm(l,m,phi,theta) 
            b = sph_harm(m,l,phi,theta)
            print(l,m, amax(abs(a-b)))
