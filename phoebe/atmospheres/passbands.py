"""
Interface to information on a variety of passbands.

.. autosummary::

    get_response
    add_response
    list_response
    subdivide_response
    eff_wave
    get_info
    get_passband_from_wavelength
    
    

Section 1. Available response functions
=======================================

Short list of available systems:

>>> import pylab as pl
>>> responses = list_response('*.*')
>>> systems = [response.split('.')[0] for response in responses]
>>> set_responses = sorted(set([response.split('.')[0] 
...                                for response in systems]))
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

+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_2MASS.png           | .. image:: images/atmospheres_passbands_ACSHRC.png          | .. image:: images/atmospheres_passbands_ACSSBC.png          |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_ACSWFC.png          | .. image:: images/atmospheres_passbands_AKARI.png           | .. image:: images/atmospheres_passbands_ANS.png             |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_ARGUE.png           | .. image:: images/atmospheres_passbands_BESSEL.png          | .. image:: images/atmospheres_passbands_COROT.png           |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_COUSINS.png         | .. image:: images/atmospheres_passbands_DENIS.png           | .. image:: images/atmospheres_passbands_DIRBE.png           |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_ESOIR.png           | .. image:: images/atmospheres_passbands_GAIA.png            | .. image:: images/atmospheres_passbands_GALEX.png           |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+

+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_GENEVA.png          | .. image:: images/atmospheres_passbands_HIPPARCOS.png       | .. image:: images/atmospheres_passbands_IPHAS.png           |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_IRAC.png            | .. image:: images/atmospheres_passbands_IRAS.png            | .. image:: images/atmospheres_passbands_ISOCAM.png          |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_JOHNSON.png         | .. image:: images/atmospheres_passbands_KEPLER.png          | .. image:: images/atmospheres_passbands_KRON.png            |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_LANDOLT.png         | .. image:: images/atmospheres_passbands_MIPS.png            | .. image:: images/atmospheres_passbands_MOST.png            |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_MSX.png             | .. image:: images/atmospheres_passbands_NARROW.png          | .. image:: images/atmospheres_passbands_NICMOS.png          |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+

+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_PACS.png            | .. image:: images/atmospheres_passbands_SAAO.png            | .. image:: images/atmospheres_passbands_SCUBA.png           |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_SDSS.png            | .. image:: images/atmospheres_passbands_SLOAN.png           | .. image:: images/atmospheres_passbands_SPIRE.png           |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_STISCCD.png         | .. image:: images/atmospheres_passbands_STISFUV.png         | .. image:: images/atmospheres_passbands_STISNUV.png         |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_STROMGREN.png       | .. image:: images/atmospheres_passbands_TD1.png             | .. image:: images/atmospheres_passbands_TYCHO.png           |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_TYCHO2.png          | .. image:: images/atmospheres_passbands_ULTRACAM.png        | .. image:: images/atmospheres_passbands_USNOB1.png          |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+

+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_VILNIUS.png         | .. image:: images/atmospheres_passbands_VISIR.png           | .. image:: images/atmospheres_passbands_WALRAVEN.png        |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| .. image:: images/atmospheres_passbands_WFPC2.png           | .. image:: images/atmospheres_passbands_WISE.png            | .. image:: images/atmospheres_passbands_WOOD.png            |
|    :width: 300px                                            |    :width: 300px                                            |    :width: 300px                                            |
+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+

Section 2. File format
=======================================

PHOEBE uses a standard library of passbands (see above), which can be complemented by a user-defined
library. The passbands should be provided in a ``.fits`` file. 

Each FITS table specifies one passband and has the following header keys (KEY (type)):

    - C{SYSTEM} (string): filter system (e.g. 'Johnson')
    - C{PASSBAND} (string): passband (e.g. 'V')
    - C{SOURCE} (string): reference to the passband measurement/definition (optional)
    - C{VERSION} (integer): version of the form 1.0, (optional, 1 is used when not specified)
    - C{EXTNAME} (string): Unique name of the extension, recommended form: JOHNSON.V.1
    - C{DETTYPE} (string): Type of detector, 'flux' (e.g. for a CCD) or 'energy' (e.g. PMT)
    - C{WAVLEFF} (float): Effective wavelength using a gray atmosphere
    - C{COMMENTS} (string): Free comment field
    - C{ZPFLUX} (float): Flux zeropoint
    - C{ZPFLUXUN} (string): Unit of the flux zeropoint (must be 'erg/s/cm2/AA' or 'erg/s/cm2/Hz')
    - C{ZPMAG} (float): Magnitude zeropoint
    - C{ZPMAGTYP} (string): Type of magnitude zeropoint (must be 'Vega', 'AB' or 'ST')

Each FITS table has a data array with the following columns:

    - C{WAVELENGTH}: wavelength array (Angstrom)
    - C{RESPONSE}: response (only relative values are needed, absolute scaling is irrelevant)
    

**Example**::

    XTENSION= 'BINTABLE'           / binary table extension                         
    BITPIX  =                    8 / array data type                                                                                                                                 
    NAXIS   =                    2 / number of array dimensions                                                                                                                      
    NAXIS1  =                   76 / length of dimension 1                                                                                                                           
    NAXIS2  =                 1221 / length of dimension 2                                                                                                                           
    PCOUNT  =                    0 / number of group parameters                                                                                                                      
    GCOUNT  =                    1 / number of groups                                                                                                                                
    TFIELDS =                   19 / number of table fields                                                                                                                          
    TTYPE1  = 'wavelength'                                                          
    TFORM1  = 'E       '                                                            
    TTYPE2  = 'response'                                                            
    TFORM2  = 'E       '                                                            
    EXTNAME = 'JOHNSON.V.1'     / name of the extension                          
    SYSTEM  = 'Johnson'        / Filter system
    PASSBAND= 'V'              / Passband
    SOURCE  = 'Johnson1953'    / Source of the response curve
    VERSION = '1'              / Version number
    DETTYPE = 'flux'           / Detector type
    WAVLEFF = '4567.2'         / Effective wavelength
    COMMENTS= 'Not needed'     / Comments
    ZPFLUX  = '1e-8'           / Zeropoint flux
    ZPFLUXUN= 'erg/s/cm2/AA'   / Flux unit
    ZPMAG   = '0.03'           / Zeropoint magnitude
    ZPMAGTYP= 'Vega'           / Vega, AB, ST  
    
"""
import numpy as np
import os
import glob
import logging
import pyfits as pf
from phoebe.utils import decorators

# Allow for on-the-fly addition of photometric passbands.
custom_passbands = {'_prefer_file': False}

logger = logging.getLogger('ATM.PB')

#    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'ptf', '*.fits')))
    
#    for filename in files:
#        fitsf = pf.open(filename)
#        for response in fitsf[0:]:
#            wavl = response.data['WAVELENGTH']
#            resp = response.data['RESPONSE']
#            tag = response.header['EXTNAME']
#            add_response(wavl, resp, passband=tag, force=True)
#        fitsf.close()

@decorators.memoized
def get_response(passband,full_output=False):
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
    @return: (wavelength [A], response) or (wavelength [A], response, header)
    @rtype: (array, array) or (array, array, dict)
    """
    passband = passband.upper()
    prefer_file = custom_passbands['_prefer_file']

    avail = list_response(passband,files_only=True)
    
    # It's easy for a bolometric filter: totally open
    if passband == 'OPEN.BOL':
        outp = np.array([1, 1e10]), np.array([1 / (1e10-1), 1 / (1e10-1)]), dict(dettype='flux', WAVLEFF=1000)
    
    # Either get the passband from a file or get it from the dictionary of
    # custom passbands
    
    # If in files files have preference over custom passbands:
    elif len(avail)==1 and prefer_file:
        outp = get_response_from_files(avail[0])
    # Else, if the custom filter exists and is preferred
    elif passband in custom_passbands:
        outp = custom_passbands[passband]
    # Else, if the file is not preferred but a custom one does not exist:
    elif len(avail)==1:
        outp = get_response_from_files(avail[0])
    # Else, if custom is not preffered and there are more than one matching filters in the file(s):
    elif len(avail)>1:
        # If there is an exact match discarding the version number, we take that one.
        # This allows to, e.g., request geneva.b and not get confusion with geneva.b1_v1.0
        avail_ = list_response(passband,files_only=True,perfect_name_match=True)
        if len(avail_)==1:
            outp = get_response_from_files(avail_[0],)        
        else:
            raise IOError('{0} exists in multiple versions: {1}'.format(passband,avail)) 
    # Else we don't know what to do
    else:
        raise IOError('{0} does not exist {1}: perhaps you need to run create_passbandfits.py inside phoebe/devel/atmospheres/?'.format(passband,custom_passbands.keys())) 

    
    # make sure the contents are sorted according to wavelength
    sort_array = np.argsort(outp[0])
    
    if full_output:
        return outp
    else:
        return outp[0:2]
    


def get_response_from_files(passband):
    """Get wave and resp and header for a passband from the fits files. Only one passband with the given name
    should exist!"""
    # File based responses
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'ptf', '*.fits')))
    for filename in files:
        try: 
            fitsf = pf.open(filename)
            ext = fitsf[passband]
            wave = ext.data['wavelength']
            resp = ext.data['response']
            head = ext.header
        finally:
            fitsf.close()
    
    return wave, resp, head

@decorators.memoized
def get_response_OLD(passband):
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
    prefer_file = custom_passbands['_prefer_file']
    
    # It's easy for a bolometric filter: totally open
    if passband == 'OPEN.BOL':
        wave, response = np.array([1, 1e10]), np.array([1 / (1e10-1), 1 / (1e10-1)])
        
    
    # Either get the passband from a file or get it from the dictionary of
    # custom passbands
    photfile = os.path.join(os.path.dirname(__file__), 'ptf', passband)
    photfile_is_file = os.path.isfile(photfile)
    
    # If the file exists and files have preference over custom passbands:
    if photfile_is_file and prefer_file:
        wave, response = np.loadtxt(photfile, unpack=True)[:2]
    
    # Else, if the custom filter exists and is preferred
    elif passband in custom_passbands:
        wave, response = custom_passbands[passband]['response']
    
    # Else, if the file is not preferred but a custom one does not exist:
    elif photfile_is_file:
        wave, response = np.loadtxt(photfile, unpack=True)[:2]
    
    # Else we don't know what to do
    else:
        raise IOError('{0} does not exist {1}'.format(passband,custom_filters.keys())) 
    
    # make sure the contents are sorted according to wavelength
    sort_array = np.argsort(wave)
    return wave[sort_array], response[sort_array]

   
def read_standard_response(filename):
    """
    Read standardized format of passbands.
    
    :param filename: absolute filepath of the passband file
    :type filename: str
    :return: (wavelength (angstrom), transmission), meta-data
    :rtype: (array, array), dict
    """
    
    # Predefined meta-data fields
    fields = ['PASS_SET', 'PASSBAND', 'EFFWL', 'WLFACTOR', 'REFERENCE']
    header = dict()
    
    with open(filename, 'r') as ff:
        while True:
            # only read the header
            line = ff.readline().strip()
            if not line:
                break
            if not line[0] == '#':
                break
            
            # read the header information
            field, entry = line[1:].split(None,1)
            if field in fields:
                header[field] = entry
                fields.remove(field)
        
    # Every response file needs to have all fields.    
    if len(fields):
        raise ValueError("Not a standard response file")
    
    data = np.loadtxt(filename).T
    
    return data, header



def set_standard_response():
    """
    Collect info on passbands.
    

    
    """
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'ptf', '*')))
    
    for filename in files:
        try:
            data, header = read_standard_response(filename)
        except ValueError:
            continue
        
        tag = '{}.{}'.format(header['PASS_SET'].upper(), header['PASSBAND'].upper())
        add_response(data[0], data[1], passband=tag, force=True)
    
    custom_passbands['_prefer_file'] = False


def eff_wave(passband, model=None, det_type=None):
    r"""
    Return the effective wavelength of a photometric passband.
    
    The effective wavelength is defined as the pivot wavelength [Tokunaga2005].
    See eff_wave_arrays() for the definition.
    
    @param passband: photometric passband
    @type passband: str ('SYSTEM.FILTER') or array/list of str
    @param model: model wavelength and fluxes
    @type model: tuple of 1D arrays (wave,flux)
    @param det_type: detector type
    @type det_type: ``BOL`` or ``CCD``
    @return: effective wavelength [A]
    @rtype: float or numpy array
    """
    
    # If passband is a string, it's the name of a passband: put it in a
    # container but unwrap afterwards
    if isinstance(passband, str):
        single_band = True
        passband = [passband]
        
    # Else, it is a container
    else:
        single_band = False
       
    # Then run over each passband and compute the effective wavelength   
    my_eff_wave = []
    for ipassband in passband:
        try:
            wave, response = get_response(ipassband)
            
            #-- bolometric or ccd?
            if det_type is None and len(get_info([ipassband])):
                det_type = get_info([ipassband])['type'][0]
            elif det_type is None:
                det_type = 'CCD'

        # If the passband is not defined, set the effective wavelength to nan
        except IOError:
            logger.error("Cannot find passband {}".format(ipassband))
            this_eff_wave = np.nan

        # Call the function that computes the effwavelength from arrays
        this_eff_wave = eff_wave_arrays(resparray=(wave,response),model=model,det_type=det_type)
        
        # Remember the result    
        my_eff_wave.append(this_eff_wave)
    
    # And provide an intuitive return value: if the user gave a single passband,
    # return a single wavelength. Otherwise return an array
    if single_band:
        my_eff_wave = my_eff_wave[0]
    else:
        my_eff_wave = np.array(my_eff_wave, float)
    
    return my_eff_wave

def eff_wave_arrays(resparray, model=None, det_type=None):
    r"""
    Return the effective wavelength of a photometric passband given the wavelength and response
    arrays. This is called by eff_wave() which works starting from a filter name.
    
    The effective wavelength is defined as the pivot wavelength [Tokunaga2005]_
    of the passband :math:`P` (for photon counting devices):
    
    .. math::
    
        \lambda_\mathrm{pivot} = \sqrt{ \frac{\int_\lambda \lambda P(\lambda) d\lambda}{ \int_\lambda P(\lambda)/\lambda d\lambda}}
    
    >>> eff_wave('2MASS.J')
    12393.093155655277
    
    If you give model fluxes as an extra argument, the wavelengths will take
    these into account to calculate the true effective wavelength (e.g., 
    [VanDerBliek1996]_, Eq 2.)
    
    @param resparray: passband defition (arrays)
    @type resparray: type of 1D arrays (wave, response)
    @param model: model wavelength and fluxes
    @type model: tuple of 1D arrays (wave,flux)
    @param det_type: detector type
    @type det_type: ``BOL`` or ``CCD``
    @return: effective wavelength [A]
    @rtype: float
    """

    # If the passband is not defined, set it to nan, otherwise compute the
    # effective wavelength.
    try:
        wave, response = resparray
        
        #-- bolometric or ccd?
        if det_type is None:
            det_type = 'CCD'
            
        # If a model is given take it into account
        if model is None:
            if det_type == 'BOL':
                this_eff_wave = np.sqrt(np.trapz(response,x=wave)/np.trapz(response/wave**2,x=wave))
            else:
                this_eff_wave = np.sqrt(np.trapz(wave*response,x=wave)/np.trapz(response/wave,x=wave))

        else:
            # Interpolate response curve onto higher resolution model and
            # take weighted average
            #is_response = response > 1e-10
            #start_response = wave[is_response].min()
            #end_response = wave[is_response].max()
            fluxm = np.sqrt(10**np.interp(np.log10(wave),np.log10(model[0]),np.log10(model[1])))
            
            if det_type=='CCD':
                this_eff_wave = np.sqrt(np.trapz(wave*fluxm*response,x=wave) / np.trapz(fluxm*response/wave,x=wave))
            elif det_type=='BOL':
                this_eff_wave = np.sqrt(np.trapz(fluxm*response,x=wave) / np.trapz(fluxm*response/wave**2,x=wave))     
    
    # If the passband is not defined, set the effective wavelength to nan
    except IOError:
        this_eff_wave = np.nan
    return this_eff_wave


def list_response(name='',files_only=False, perfect_name_match=False):
    """
    List available response curves matching C{name}.
    
    Files_only allows to only use the filters defined in files.
    Perfect_name_match allows to search for perfect matches of filter name, only leaving version
    numbers free.
    
    Example usage:
    
    >>> print(list_response('GENEVA.'))
    ['GENEVA.B', 'GENEVA.B1', 'GENEVA.B2', 'GENEVA.G', 'GENEVA.U', 'GENEVA.V', 'GENEVA.V1']
    
    @param name: string matching response curve name.
    @type name: str
    @param files_only: only use files or also custom passbands
    @type files_only: boolean
    @return: list of responses
    @rtype: list of str

    """
    # File based responses
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'ptf', '*.fits')))
    responses = []
    for filename in files:
        fitsf = pf.open(filename)
        for response in fitsf[1:]:
            responses.append(response.header['EXTNAME'])
        fitsf.close()
    
    responses = sorted(responses)
    
    # Custom responses
    if not files_only:
        responses = sorted(responses + \
               [key for key in custom_passbands.keys() \
                          if ((name in key) and not (key=='_prefer_file'))])
    
    if perfect_name_match:
        return [resp_ for resp_ in responses if name.lower() == resp_.lower().split('_v')[0]]
    else:
        return [resp_ for resp_ in responses if name.lower() in resp_.lower()]

    
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
    
    @param passbands: list of passbands to get the information from.
      The input order is equal to the output order. If C{None}, all filters are
      returned.
    @type passbands: iterable container (list, tuple, 1Darray)
    @return: record array containing all information on the requested passbands.
    @rtype: record array
    """
    # Read in the file
    zp_file = os.path.join(os.path.dirname(__file__), 'ptf', 'zeropoints.dat')
    zp_table = np.loadtxt(zp_file, dtype=str, unpack=True)
    
    # Define column name and type
    header = ['passband', 'eff_wave', 'type', 'transmission', 'vegamag',
              'vegamag_lit', 'ABmag', 'ABmag_lit', 'STmag', 'STmag_lit',
              'Flam0', 'Flam0_units', 'Flam0_lit', 'Fnu0', 'Fnu0_units',
              'Fnu0_lit', 'source']
    types = ['S50', '>f8', 'S3', '>f8', '>f8', 'int32', '>f8', 'int32', '>f8',
             'int32', '>f8', 'S50', 'int32', '>f8', 'S50', 'int32', 'S100']
    dtype = [(head, typ) for head, typ in zip(header, types)]
    dtype = np.dtype(dtype)
    
    # Cast the columns to the right type
    zp_table = [np.cast[dtype[i]](zp_table[i]) for i in range(len(zp_table))]
    zp_table = np.rec.array(zp_table, dtype=dtype)
    #print zp_table
    zp_extra = [custom_passbands[key]['zp'] for key in custom_passbands if not key=='_prefer_file' and 'zp' in custom_passbands[key]]
    zp_table = np.hstack([zp_table] + zp_extra)
    zp_table = zp_table[np.argsort(zp_table['passband'])]
    
    # List passbands in order given, and remove those that do not have
    # zeropoints etc.
    if passbands is not None:
        order = np.searchsorted(zp_table['passband'], passbands)
        zp_table = zp_table[order]
        keep = (zp_table['passband'] == passbands)
        zp_table = zp_table[keep]
    
    # That's it
    return zp_table


def get_passband_from_wavelength(wavelength, wrange=100.0):
    """
    Retrieve the passbands in the vicinity around a central wavelength.
    
    Return the closest one and all in a certain range.
    """
    
    # Retrieve all passband information
    info = get_info()
    
    best_index = np.argmin(np.abs(info['eff_wave'] - wavelength))
    valid = np.abs(info['eff_wave']-wavelength) <= (wrange/2.0)
    
    return info[best_index]['passband'], info[valid]['passband']



def add_response(wave, response, passband='CUSTOM.PTF', force=False,
                 copy_from='JOHNSON.V', **kwargs):
    """
    Add a custom passband to the set of predefined passbands.
    
    Extra keywords are:
        'eff_wave', 'type',
        'vegamag', 'vegamag_lit',
        'ABmag', 'ABmag_lit',
        'STmag', 'STmag_lit',
        'Flam0', 'Flam0_units', 'Flam0_lit',
        'Fnu0', 'Fnu0_units', 'Fnu0_lit',
        'source'
        
    default ``type`` is ``'CCD'``.
    default ``passband`` is ``'CUSTOM.PTF'``
    
    @param wave: wavelength (angstrom)
    @type wave: ndarray
    @param response: response
    @type response: ndarray
    @param passband: photometric passband
    @type passband: str (``'SYSTEM.FILTER'``)
    """
    # Clear all things that are memoized from this module
    decorators.clear_memoization(keys=['passbands'])
    
    # Check if the passband already exists:
    photfile = os.path.join(os.path.dirname(__file__), 'ptf', passband)
    if os.path.isfile(photfile) and not force:
        raise ValueError('bandpass {0} already exists'.format(photfile))
    elif passband in custom_passbands:
        logger.debug('Overwriting previous definition of {0}'.format(passband))
    
    # Set effective wavelength
    kwargs.setdefault('dettype', 'flux')
    kwargs.setdefault('WAVLEFF', eff_wave_arrays((wave,response), det_type=kwargs['dettype']))
    
    # Add info for zeropoints.dat file: make sure wherever "lit" is part of the
    # name, we replace it with "0". Then, we overwrite any existing information
    # with info given
    myrow = get_response(copy_from, full_output=True)[2]
    for kwarg in kwargs.keys():
        myrow[kwarg] = kwargs[kwarg]
    #for name in myrow.dtype.names:
    #    if 'lit' in name:
    #        myrow[name] = 0
    #    myrow[name] = kwargs.pop(name, myrow[name])
    myrow['passband'] = passband
        
    # Remove memoized things to be sure to have the most up-to-date info
    del decorators.memory[__name__]
    
    # Finally add the info:
    custom_passbands[passband] = wave, response, myrow
    logger.info('Added passband {0} to the predefined set'.format(passband))


def subdivide_response(passband, parts=10, sampling=100, add=True):
    """
    Subdivide a passband in a number of narrower parts.
    
    This can be useful to use in interferometric bandwidth smearing.
    
    If ``add=True``, a new passband will be added with the naming scheme::
    
        SYSTEM.FILTER_<PART>_<TOTAL_PARTS>
        
    @param passband: passband to subdivide
    @type passband: str
    """
    
    # Get the original response
    wave, response = get_response(passband)
    
    # Concentrate on the part that isn't zero:
    positive = np.array(response > 0, int)
    edges = np.diff(positive)
    start = np.argmax(edges)
    end = len(edges) - np.argmin(edges[::-1])
    
    # Divide the response in the number of parts
    new_edges = np.linspace(wave[start], wave[end], parts+1)
    responses = []
    
    for part in range(parts):
        
        # Create a new wavelength array for this part of the response curve
        this_wave = np.linspace(new_edges[part], new_edges[part+1], sampling)
        delta_wave = this_wave[1] - this_wave[0]
    
        # And extend it slightly to have really sharp edges.
        this_wave = np.hstack([this_wave[0]-delta_wave/100.,
                               this_wave,
                               this_wave[-1]+delta_wave/100.])
        
        # Interpolate the response curve on this part, and make edges sharp
        this_response = np.interp(this_wave, wave, response)
        this_response[0] = 0.
        this_response[-1] = 0.
        responses.append((this_wave,this_response))
        
        # Give them a default name, which the user can use to refer to them
        # (but doesn't need to)
        new_name = "{}_{:04d}_{:04d}".format(passband, part, parts)
        names.append(new_name)
        
        if add:
            add_response(this_wave, this_response, passband=new_name,
                         force=True, copy_from=passband)
        
    return responses, names
    

def create_passbands_fitsfile(filelist):
    None
    
    
    
    

if __name__ == "__main__":
    import doctest
    doctest.testmod()