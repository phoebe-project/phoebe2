"""
Interface to information on a variety of passbands.

.. autosummary::

    get_response
    list_response
    eff_wave
    get_info
    

Section 1. Available response functions
=======================================

Short list of available systems:

>>> import pylab as pl
>>> responses = list_response('*.*')
>>> systems = [response.split('.')[0] for response in responses]
>>> set_responses = sorted(set([response.split('.')[0] for response in systems]))
>>> for i,resp in enumerate(set_responses):
...    print '%10s (%3d filters)'%(resp,systems.count(resp))
     2MASS (  3 filters)
    ACSHRC ( 17 filters)
    ACSSBC (  6 filters)
    ACSWFC ( 12 filters)
     AKARI ( 13 filters)
       ANS (  6 filters)
     ARGUE (  3 filters)
    BESSEL (  6 filters)
     COROT (  2 filters)
   COUSINS (  3 filters)
     DENIS (  3 filters)
     DIRBE ( 10 filters)
     ESOIR ( 10 filters)
      GAIA (  4 filters)
     GALEX (  2 filters)
    GENEVA (  7 filters)
 HIPPARCOS (  1 filters)
     IPHAS (  3 filters)
      IRAC (  4 filters)
      IRAS (  4 filters)
    ISOCAM ( 21 filters)
   JOHNSON ( 25 filters)
    KEPLER ( 43 filters)
      KRON (  2 filters)
   LANDOLT (  6 filters)
      MIPS (  3 filters)
      MOST (  1 filters)
       MSX (  6 filters)
    NARROW (  1 filters)
    NICMOS (  6 filters)
      PACS (  3 filters)
      SAAO ( 13 filters)
     SCUBA (  6 filters)
      SDSS ( 10 filters)
     SLOAN (  2 filters)
     SPIRE (  3 filters)
   STISCCD (  2 filters)
   STISFUV (  4 filters)
   STISNUV (  7 filters)
 STROMGREN (  6 filters)
       TD1 (  4 filters)
     TYCHO (  2 filters)
    TYCHO2 (  2 filters)
  ULTRACAM (  5 filters)
    USNOB1 (  2 filters)
   VILNIUS (  7 filters)
     VISIR ( 13 filters)
  WALRAVEN (  5 filters)
     WFCAM (  5 filters)
     WFPC2 ( 21 filters)
      WISE (  4 filters)
      WOOD ( 12 filters)

Plots of all passbands of all systems:

]include figure]]images/atmospheres_passbands_2MASS.png]

]include figure]]images/atmospheres_passbands_ACSHRC.png]

]include figure]]images/atmospheres_passbands_ACSSBC.png]

]include figure]]images/atmospheres_passbands_ACSWFC.png]

]include figure]]images/atmospheres_passbands_AKARI.png]

]include figure]]images/atmospheres_passbands_ANS.png]

]include figure]]images/atmospheres_passbands_ARGUE.png]

]include figure]]images/atmospheres_passbands_BESSEL.png]

]include figure]]images/atmospheres_passbands_COROT.png]

]include figure]]images/atmospheres_passbands_COUSINS.png]

]include figure]]images/atmospheres_passbands_DENIS.png]

]include figure]]images/atmospheres_passbands_DIRBE.png]

]include figure]]images/atmospheres_passbands_ESOIR.png]

]include figure]]images/atmospheres_passbands_GAIA.png]

]include figure]]images/atmospheres_passbands_GALEX.png]

]include figure]]images/atmospheres_passbands_GENEVA.png]

]include figure]]images/atmospheres_passbands_HIPPARCOS.png]

]include figure]]images/atmospheres_passbands_IPHAS.png]

]include figure]]images/atmospheres_passbands_IRAC.png]

]include figure]]images/atmospheres_passbands_IRAS.png]

]include figure]]images/atmospheres_passbands_ISOCAM.png]

]include figure]]images/atmospheres_passbands_JOHNSON.png]

]include figure]]images/atmospheres_passbands_KEPLER.png]

]include figure]]images/atmospheres_passbands_KRON.png]

]include figure]]images/atmospheres_passbands_LANDOLT.png]

]include figure]]images/atmospheres_passbands_MIPS.png]

]include figure]]images/atmospheres_passbands_MOST.png]

]include figure]]images/atmospheres_passbands_MSX.png]

]include figure]]images/atmospheres_passbands_NARROW.png]

]include figure]]images/atmospheres_passbands_NICMOS.png]

]include figure]]images/atmospheres_passbands_PACS.png]

]include figure]]images/atmospheres_passbands_SAAO.png]

]include figure]]images/atmospheres_passbands_SCUBA.png]

]include figure]]images/atmospheres_passbands_SDSS.png]

]include figure]]images/atmospheres_passbands_SLOAN.png]

]include figure]]images/atmospheres_passbands_SPIRE.png]

]include figure]]images/atmospheres_passbands_STISCCD.png]

]include figure]]images/atmospheres_passbands_STISFUV.png]

]include figure]]images/atmospheres_passbands_STISNUV.png]

]include figure]]images/atmospheres_passbands_STROMGREN.png]

]include figure]]images/atmospheres_passbands_TD1.png]

]include figure]]images/atmospheres_passbands_TYCHO.png]

]include figure]]images/atmospheres_passbands_TYCHO2.png]

]include figure]]images/atmospheres_passbands_ULTRACAM.png]

]include figure]]images/atmospheres_passbands_USNOB1.png]

]include figure]]images/atmospheres_passbands_VILNIUS.png]

]include figure]]images/atmospheres_passbands_VISIR.png]

]include figure]]images/atmospheres_passbands_WALRAVEN.png]

]include figure]]images/atmospheres_passbands_WFPC2.png]

]include figure]]images/atmospheres_passbands_WISE.png]

]include figure]]images/atmospheres_passbands_WOOD.png]
"""
import numpy as np
import os
import glob
from phoebe.utils import decorators

@decorators.memoized
def get_response(passband):
    """
    Retrieve the response curve of a photometric system 'SYSTEM.FILTER'
    
    OPEN.BOL represents a bolometric open filter.
    
    Example usage:
    
    >>> from pylab import plot,show
    >>> for band in ['J','H','KS']:
    ...    p = plot(*get_response('2MASS.%s'%(band)))
    >>> p = show()
    
    @param passband: photometric passband
    @type passband: str ('SYSTEM.FILTER')
    @return: (wavelength [A], response)
    @rtype: (array, array)
    """
    passband = passband.upper()
    if passband=='OPEN.BOL':
        return np.array([1,1e10]),np.array([1/(1e10-1),1/(1e10-1)])
        
    photfile = os.path.join(os.path.dirname(__file__),'ptf',passband)
    wave, response = np.loadtxt(photfile,unpack=True)[:2]
    sa = np.argsort(wave)
    return wave[sa],response[sa]

def eff_wave(passband,model=None):
    """
    Return the effective wavelength of a photometric passband.
    
    The effective wavelength is defined as the average wavelength weighed with
    the response curve.
    
    >>> eff_wave('2MASS.J')
    12412.136241640892
    
    If you give model fluxes as an extra argument, the wavelengths will take
    these into account to calculate the `true' effective wavelength (e.g., 
    Van Der Bliek, 1996), eq 2.
    
    @param passband: photometric passband
    @type passband: str ('SYSTEM.FILTER') or array/list of str
    @param model: model wavelength and fluxes
    @type model: tuple of 1D arrays (wave,flux)
    @return: effective wavelength [A]
    @rtype: float or numpy array
    """
    
    #-- if passband is a string, it's the name of a passband: put it in a container
    #   but unwrap afterwards
    if isinstance(passband,str):
        single_band = True
        passband = [passband]
    #-- else, it is a container
    else:
        single_band = False
        
    my_eff_wave = []
    for ipassband in passband:
        try:
            wave,response = get_response(ipassband)
            if model is None:
                this_eff_wave = np.average(wave,weights=response)
            else:
                #-- interpolate response curve onto higher resolution model and
                #   take weighted average
                is_response = response>1e-10
                start_response,end_response = wave[is_response].min(),wave[is_response].max()
                fluxm = 10**np.interp(np.log10(wave),np.log10(model[0]),np.log10(model[1]))
                this_eff_wave = np.trapz(wave*fluxm*response,x=wave) / np.trapz(fluxm*response,x=wave)
        #-- if the passband is not defined:
        except IOError:
            this_eff_wave = np.nan
        my_eff_wave.append(this_eff_wave)
    
    if single_band:
        my_eff_wave = my_eff_wave[0]
    else:
        my_eff_wave = np.array(my_eff_wave,float)
    
    return my_eff_wave

def list_response(name):
    """
    List available response curves matching C{name}.
    
    Example usage:
    
    >>> print(list_response('GENEVA.*'))
    ['GENEVA.B', 'GENEVA.B1', 'GENEVA.B2', 'GENEVA.G', 'GENEVA.U', 'GENEVA.V', 'GENEVA.V1']
    
    @param name: string matching response curve name.
    @type name: str
    @return: list of responses
    @rtype: list of str
    """
    responses = sorted(glob.glob(os.path.join(os.path.dirname(__file__),'ptf',name)))
    return [os.path.basename(resp) for resp in responses]
    
@decorators.memoized
def get_info(passbands=None):
    """
    Return a record array containing all filter information.
    
    The record arrays contains following columns:
        - passband
        - eff_wave
        - type
        - vegamag, vegamag_lit
        - ABmag, ABmag_lit
        - STmag, STmag_lit
        - Flam0, Flam0_units, Flam0_lit
        - Fnu0, Fnu0_units, Fnu0_lit,
        - source
    
    @param passbands: list of passbands to get the information from. The input
    order is equal to the output order. If C{None}, all filters are returned.
    @type passbands: iterable container (list, tuple, 1Darray)
    @return: record array containing all information on the requested passbands.
    @rtype: record array
    """
    zp_file = os.path.join(os.path.dirname(__file__),'ptf','zeropoints.dat')
    zp = np.loadtxt(zp_file,dtype=str,unpack=True)
    header = ['passband','eff_wave','type','vegamag','vegamag_lit','ABmag',
              'ABmag_lit','STmag','STmag_lit','Flam0','Flam0_units','Flam0_lit',
              'Fnu0','Fnu0_units','Fnu0_lit','source']
    types = ['S50','>f8','S3','>f8','int32','>f8','int32','>f8','int32','>f8',
             'S50','int32','>f8','S50','int32','S100']
    dtype = [(head,typ) for head,typ in zip(header,types)]
    dtype = np.dtype(dtype)
    zp = [np.cast[dtype[i]](zp[i]) for i in range(len(zp))]
    zp = np.rec.array(zp,dtype=dtype)
    zp = zp[np.argsort(zp['passband'])]
    
    #-- list passbands in order given, and remove those that do not have
    #   zeropoints etc.
    if passbands is not None:
        order = np.searchsorted(zp['passband'],passbands)
        zp = zp[order]
        keep = (zp['passband']==passbands)
        zp = zp[keep]
    
    return zp





if __name__=="__main__":
    import doctest
    fails,tests = doctest.testmod()
    if not fails:
        print(("All {0} tests succeeded".format(tests)))
    else:
        print(("{0}/{1} tests failed".format(fails,tests)))
