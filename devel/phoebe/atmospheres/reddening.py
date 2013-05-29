# -*- coding: utf-8 -*-
"""
Definitions of interstellar reddening curves

Section 1. General interface
============================

Use the general interface to get different curves:

>>> wave = np.r_[1e3:1e5:10]
>>> for name in ['chiar2006','fitzpatrick1999','fitzpatrick2004','cardelli1989','seaton1979']:
...   wave_,mag_ = get_law(name,wave=wave)
...   p = pl.plot(1e4/wave_,mag_,label=name)
>>> p = pl.xlim(0,10)
>>> p = pl.ylim(0,12)

Use the general interface to get the same curves but with different Rv:

>>> for Rv in [2.0,3.1,5.1]:
...   wave_,mag_ = get_law('cardelli1989',wave=wave,Rv=Rv)
...   p = pl.plot(1e4/wave_,mag_,'--',lw=2,label='cardelli1989 Rv=%.1f'%(Rv))
>>> p = pl.xlim(0,10)
>>> p = pl.ylim(0,12)
>>> p = pl.xlabel('1/$\lambda$ [1/$\mu$m]')
>>> p = pl.ylabel(r'Extinction E(B-$\lambda$) [mag]')
>>> p = pl.legend(prop=dict(size='small'),loc='lower right')

]include figure]]images/atmospheres_reddening_curves.png]

Section 2. Individual curve definitions
=======================================

Get the curves seperately:

>>> wave1,mag1 = cardelli1989()
>>> wave2,mag2 = chiar2006()
>>> wave3,mag3 = seaton1979()
>>> wave4,mag4 = fitzpatrick1999()
>>> wave5,mag5 = fitzpatrick2004()


Section 3. Normalisations
=========================

>>> wave = np.logspace(3,6,1000)
>>> passbands = ['JOHNSON.V','JOHNSON.K']

Retrieve two interstellar reddening laws, normalise them to Av and see what
the ratio between Ak and Av is. Since the L{chiar2006} law is not defined
in the optical, this procedure actually doesn't make sense for that law. In the
case of L{cardelli1989}, compute the ratio C{Ak/Av}. Note that you cannot
ask for the L{chiar2006} to be normalised in the visual, since the curve is
not defined there! If you really want to do that anyway, you need to derive
a Ak/Av factor from another curve (e.g. the L{cardelli1989}).

>>> p = pl.figure()
>>> for name,norm in zip(['chiar2006','cardelli1989'],['JOHNSON.K','JOHNSON.V']):
...   wave_,mag_ = get_law(name,wave=wave,Aband=norm)
...   photwave,photflux = get_law(name,wave=wave,Aband=norm,passbands=passbands)
...   p = pl.plot(wave_/1e4,mag_,label=name)
...   p = pl.plot(photwave/1e4,photflux,'s')
...   if name=='cardelli1989': print 'Ak/Av = %.3g'%(photflux[1]/photflux[0])
Ak/Av = 0.114
>>> p = pl.gca().set_xscale('log')
>>> p = pl.gca().set_yscale('log')
>>> p = pl.xlabel('Wavelength [micron]')
>>> p = pl.ylabel('Extinction A($\lambda$)/Av [mag]')

]include figure]]images/atmospheres_reddening_02.png]

Compute the Cardelli law normalised to Ak and Av.

>>> p = pl.figure()
>>> wave_av1,mag_av1 = get_law('cardelli1989',wave=wave,Aband='JOHNSON.V')
>>> wave_av2,mag_av2 = get_law('cardelli1989',wave=wave,Aband='JOHNSON.V',passbands=['JOHNSON.V'])
>>> p = pl.plot(wave_av1,mag_av1,'k-',lw=2,label='Av')
>>> p = pl.plot(wave_av2,mag_av2,'ks')

>>> wave_ak1,mag_ak1 = get_law('cardelli1989',wave=wave,Aband='JOHNSON.K')
>>> wave_ak2,mag_ak2 = get_law('cardelli1989',wave=wave,Aband='JOHNSON.K',passbands=['JOHNSON.K'])
>>> p = pl.plot(wave_ak1,mag_ak1,'r-',lw=2,label='Ak')
>>> p = pl.plot(wave_ak2,mag_ak2,'rs')

>>> p = pl.gca().set_xscale('log')
>>> p = pl.gca().set_yscale('log')
>>> p = pl.xlabel('Wavelength [micron]')
>>> p = pl.ylabel('Extinction A($\lambda$)/A($\mu$) [mag]')

]include figure]]images/atmospheres_reddening_03.png]
"""

import os
import numpy as np
import logging
from phoebe.utils import decorators
from phoebe.atmospheres import sed
from phoebe.atmospheres import passbands as pbs

logger = logging.getLogger("SED.RED")
logger.addHandler(logging.NullHandler())

basename = os.path.join(os.path.dirname(__file__),'redlaws')

#{ Main interface

def get_law(name,passbands=None,**kwargs):
    """
    Retrieve an interstellar reddening law.
    
    Parameter C{name} must be the function name of one of the laws defined in
    this module.
    
    By default, the law will be interpolated on a grid from 100 angstrom to
    10 :math:`\mu`m in steps of 10 :math:`\AA`. This can be adjusted with the parameter
    C{wave} (array), which B{must} be in angstrom. You can change the units
    ouf the returned wavelength array via C{wave_units}.
    
    By default, the curve is normalised with respect to E(B-V) (you get
    :math:`A(\lambda)/E(B-V)`). You can set the C{norm} keyword to :math:`A_V`
    if you want :math:`A(\lambda)/A_V`.
    
    Remember that
    
    ..math::
        
        A(V) = R_V E(B-V)
    
    The parameter :math:`R_V` is by default 3.1, other reasonable values lie between
    2.0 and 5.1
    
    Extra accepted keywords depend on the type of reddening law used.
    
    Example usage:
    
    >>> wave = np.r_[1e3:1e5:10]
    >>> wave,mag = get_law('cardelli1989',wave=wave,Rv=3.1)
    
    @param name: name of the interstellar law
    @type name: str, one of the functions defined here
    @param passbands: list of photometric passbands
    @type passbands: list of strings
    @param wave: wavelength array to interpolate the law on
    @type wave: array
    @return: wavelength, reddening magnitude
    @rtype: (array,array)
    """
    #-- get the inputs
    wave_ = kwargs.pop('wave',None)
    Rv = kwargs.setdefault('Rv',3.1)
    Aband = kwargs.pop('Aband',None) # normalisation band if A is given.
    
    #-- get the curve
    wave,mag = globals()[name.lower()](**kwargs)
    wave_orig,mag_orig = wave.copy(),mag.copy()
    
    #-- interpolate on user defined grid
    if wave_ is not None:
        mag = np.interp(wave_,wave,mag,right=0)
        wave = wave_
           
    #-- pick right normalisation: convert to A(lambda)/Av if needed
    if Aband is None:
        mag *= Rv
    else:
        norm_reddening = sed.synthetic_flux(wave_orig,mag_orig,[Aband])[0]
        logger.info('Normalisation via %s: Av/%s = %.6g'%(Aband,Aband,1./norm_reddening))
        mag /= norm_reddening
    
    #-- maybe we want the curve in photometric pbs
    if passbands is not None:
        mag = sed.synthetic_flux(wave,mag,passbands)
        wave = pbs.get_info(passbands)['eff_wave']
    
    return wave,mag


def redden(flux,wave=None,passbands=None,ebv=0.,rtype='flux',law='cardelli1989',**kwargs):
    """
    Redden flux or magnitudes
    
    The reddening parameters C{ebv} means E(B-V).
    
    If it is negative, we B{deredden}.
    
    If you give the keyword C{wave}, it is assumed that you want to (de)redden
    a B{model}, i.e. a spectral energy distribution.
    
    If you give the keyword C{passbands}, it is assumed that you want to (de)redden
    B{photometry}, i.e. integrated fluxes.
    
    @param flux: fluxes to (de)redden (magnitudes if C{rtype='mag'})
    @type flux: array (floats)
    @param wave: wavelengths matching the fluxes (or give C{passbands})
    @type wave: array (floats)
    @param passbands: photometry bands matching the fluxes (or give C{wave})
    @type passbands: array of str
    @param ebv: reddening parameter E(B-V)
    @type ebv: float
    @param rtype: type of dereddening (magnituds or fluxes)
    @type rtype: str ('flux' or 'mag')
    @return: (de)reddened flux/magnitude
    @rtype: array
    """
    if passbands is not None:
        wave = pbs.get_info(passbands)['eff_wave']
        
    wave,mag = get_law(law,wave=wave,**kwargs)
    if rtype=='flux':
        flux_dered = flux / 10**(mag*ebv/2.5)
    elif rtype=='mag':
        flux_dered = flux - mag*ebv
    return flux_dered

def deredden(flux,wave=None,passbands=None,ebv=0.,rtype='flux',**kwargs):
    """
    Deredden flux or magnitudes.
    
    @param flux: fluxes to (de)redden (NOT magnitudes)
    @type flux: array (floats)
    @param wave: wavelengths matching the fluxes (or give C{passbands})
    @type wave: array (floats)
    @param passbands: photometry bands matching the fluxes (or give C{wave})
    @type passbands: array of str
    @param ebv: reddening parameter E(B-V)
    @type ebv: float
    @param rtype: type of dereddening (magnituds or fluxes)
    @type rtype: str ('flux' or 'mag')
    @return: (de)reddened flux
    @rtype: array
    """
    return redden(flux,wave=wave,passbands=passbands,ebv=-ebv,rtype=rtype,**kwargs)
    

#}

#{ Curve definitions
@decorators.memoized
def chiar2006(Rv=3.1,curve='ism',**kwargs):
    """
    Extinction curve at infrared wavelengths from Chiar and Tielens (2006)
    
    We return A(lambda)/E(B-V), by multiplying A(lambda)/Av with Rv.
    
    This is only defined for Rv=3.1. If it is different, this will raise an
    AssertionError
    
    Extra kwags are to catch unwanted keyword arguments.
    
    UNCERTAIN NORMALISATION
    
    @param Rv: Rv
    @type Rv: float
    @param curve: extinction curve
    @type curve: string (one of 'gc' or 'ism', galactic centre or local ISM)
    @return: wavelengths (A), A(lambda)/Av
    @rtype: (array,array)
    """
    source = os.path.join(basename,'Chiar2006.red')
    
    #-- check Rv
    assert(Rv==3.1)
    wavelengths,gc,ism = np.loadtxt(source).T
    if curve=='gc':
        alam_ak = gc
    elif curve=='ism':
        keep = ism>0
        alam_ak = ism[keep]
        wavelengths = wavelengths[keep]
    else:
        raise ValueError('no curve %s'%(curve))
    alam_aV = alam_ak * 0.09
    #plot(1/wavelengths,alam_aV,'o-')
    return wavelengths*1e4,alam_aV



@decorators.memoized
def fitzpatrick1999(Rv=3.1,**kwargs):
    """
    From Fitzpatrick 1999 (downloaded from ASAGIO database)
    
    This function returns A(lambda)/A(V).
    
    To get A(lambda)/E(B-V), multiply the return value with Rv (A(V)=Rv*E(B-V))
    
    Extra kwags are to catch unwanted keyword arguments.
    
    @param Rv: Rv (2.1, 3.1 or 5.0)
    @type Rv: float
    @return: wavelengths (A), A(lambda)/Av
    @rtype: (array,array)
    """
    filename = 'Fitzpatrick1999_Rv_%.1f'%(Rv)
    filename = filename.replace('.','_') + '.red'
    myfile = os.path.join(basename,filename)
    wave,alam_ebv = np.loadtxt(myfile).T
    alam_av = alam_ebv/Rv
    
    logger.info('Fitzpatrick1999 curve with Rv=%.2f'%(Rv))
    
    return wave,alam_av

@decorators.memoized
def fitzpatrick2004(Rv=3.1,**kwargs):
    """
    From Fitzpatrick 2004 (downloaded from FTP)
    
    This function returns A(lambda)/A(V).
    
    To get A(lambda)/E(B-V), multiply the return value with Rv (A(V)=Rv*E(B-V))
    
    Extra kwags are to catch unwanted keyword arguments.
    
    @param Rv: Rv (2.1, 3.1 or 5.0)
    @type Rv: float
    @return: wavelengths (A), A(lambda)/Av
    @rtype: (array,array)
    """
    filename = 'Fitzpatrick2004_Rv_%.1f.red'%(Rv)
    myfile = os.path.join(basename,filename)
    wave_inv,elamv_ebv = np.loadtxt(myfile,skiprows=15).T
    
    logger.info('Fitzpatrick2004 curve with Rv=%.2f'%(Rv))
    
    return 1e4/wave_inv[::-1],((elamv_ebv+Rv)/Rv)[::-1]


@decorators.memoized
def donnell1994(**kwargs):
    """
    Small improvement on Cardelli 1989 by James E. O'Donnell (1994).
    
    Extra kwags are to catch unwanted keyword arguments.
    
    @keyword Rv: Rv
    @type Rv: float
    @keyword wave: wavelengths to compute the curve on
    @type wave: array
    @return: wavelengths (A), A(lambda)/Av
    @rtype: (array,array)
    """
    return cardelli1989(curve='donnell',**kwargs)


@decorators.memoized
def cardelli1989(Rv=3.1,curve='cardelli',wave=None,**kwargs):
    """
    Construct extinction laws from Cardelli (1989).
    
    Improvement in optical by James E. O'Donnell (1994)
    
    wavelengths in Angstrom!
    
    This function returns A(lambda)/A(V).
    
    To get A(lambda)/E(B-V), multiply the return value with Rv (A(V)=Rv*E(B-V))
    
    Extra kwags are to catch unwanted keyword arguments.
    
    @param Rv: Rv
    @type Rv: float
    @param curve: extinction curve
    @type curve: string (one of 'cardelli' or 'donnell')
    @param wave: wavelengths to compute the curve on
    @type wave: array
    @return: wavelengths (A), A(lambda)/Av
    @rtype: (array,array)
    """
    if wave is None:
        wave = np.r_[100.:100000.:10]
    
    all_x = 1./(wave/1.0e4)
    alam_aV = np.zeros_like(all_x)
    
    #-- infrared
    infrared = all_x<1.1
    x = all_x[infrared]
    ax = +0.574*x**1.61
    bx = -0.527*x**1.61
    alam_aV[infrared] = ax + bx/Rv
    
    #-- optical
    optical = (1.1<=all_x) & (all_x<3.3)
    x = all_x[optical]
    y = x-1.82
    if curve=='cardelli':
        ax = 1 + 0.17699*y    - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 \
               + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        bx =     1.41338*y    + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 \
               - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    elif curve=='donnell':
        ax = 1 + 0.104*y    - 0.609*y**2 + 0.701*y**3 + 1.137*y**4 \
               - 1.718*y**5 - 0.827*y**6 + 1.647*y**7 - 0.505*y**8
        bx =     1.952*y    + 2.908*y**2 - 3.989*y**3 - 7.985*y**4 \
              + 11.102*y**5 + 5.491*y**6 -10.805*y**7 + 3.347*y**8
    else:
        raise ValueError('curve %s not found'%(curve))
    alam_aV[optical] = ax + bx/Rv
    
    #-- ultraviolet
    ultraviolet = (3.3<=all_x) & (all_x<8.0)
    x = all_x[ultraviolet]
    Fax = -0.04473*(x-5.9)**2 - 0.009779*(x-5.9)**3
    Fbx = +0.21300*(x-5.9)**2 + 0.120700*(x-5.9)**3
    Fax[x<5.9] = 0
    Fbx[x<5.9] = 0
    ax = +1.752 - 0.316*x - 0.104 / ((x-4.67)**2 + 0.341) + Fax
    bx = -3.090 + 1.825*x + 1.206 / ((x-4.62)**2 + 0.263) + Fbx
    alam_aV[ultraviolet] = ax + bx/Rv
    
    #-- far UV
    fuv = 8.0<=all_x
    x = all_x[fuv]
    ax = -1.073 - 0.628*(x-8) + 0.137*(x-8)**2 - 0.070*(x-8)**3
    bx = 13.670 + 4.257*(x-8) - 0.420*(x-8)**2 + 0.374*(x-8)**3
    alam_aV[fuv] = ax + bx/Rv
    
    logger.info('%s curve with Rv=%.2f'%(curve.title(),Rv))
    
    return wave,alam_aV


@decorators.memoized
def seaton1979(Rv=3.1,wave=None,**kwargs):
    """
    Extinction curve from Seaton, 1979.
    
    This function returns A(lambda)/A(V).
    
    To get A(lambda)/E(B-V), multiply the return value with Rv (A(V)=Rv*E(B-V))
    
    Extra kwags are to catch unwanted keyword arguments.
    
    @param Rv: Rv
    @type Rv: float
    @param wave: wavelengths to compute the curve on
    @type wave: array
    @return: wavelengths (A), A(lambda)/Av
    @rtype: (array,array)
    """
    if wave is None: wave = np.r_[1000.:10000.:10]
    
    all_x = 1e4/(wave)
    alam_aV = np.zeros_like(all_x)
    
    #-- far infrared
    x_ = np.r_[1.0:2.8:0.1]
    X_ = np.array([1.36,1.44,1.84,2.04,2.24,2.44,2.66,2.88,3.14,3.36,3.56,3.77,3.96,4.15,4.26,4.40,4.52,4.64])
    fir = all_x<=2.7
    alam_aV[fir] = np.interp(all_x[fir][::-1],x_,X_,left=0)[::-1]
    
    #-- infrared
    infrared = (2.70<=all_x) & (all_x<3.65)
    x = all_x[infrared]
    alam_aV[infrared] = 1.56 + 1.048*x + 1.01 / ( (x-4.60)**2 + 0.280)
    
    #-- optical
    optical = (3.65<=all_x) & (all_x<7.14)
    x = all_x[optical]
    alam_aV[optical] = 2.29 + 0.848*x + 1.01 / ( (x-4.60)**2 + 0.280)
    
    #-- ultraviolet
    ultraviolet = (7.14<=all_x) & (all_x<=10)
    x = all_x[ultraviolet]
    alam_aV[ultraviolet] = 16.17 - 3.20*x + 0.2975*x**2
    
    logger.info('Seaton curve with Rv=%.2f'%(Rv))
    
    return wave,alam_aV/Rv

#}

if __name__=="__main__":
    import doctest
    import pylab as pl
    doctest.testmod()
    pl.show()