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

"""
import numpy as np
import os
import glob
from phoebe.utils import decorators

# Allow for on-the-fly addition of photometric passbands.
custom_passbands = {'_prefer_file': True}


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
    prefer_file = custom_passbands['_prefer_file']
    
    
    # It's easy for a bolometric filter: totally open
    if passband == 'OPEN.BOL':
        return np.array([1, 1e10]), np.array([1 / (1e10-1), 1 / (1e10-1)])
    
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
        raise IOError,('{0} does not exist {1}'.format(photband,custom_filters.keys())) 
    
    # make sure the contents are sorted according to wavelength
    sort_array = np.argsort(wave)
    return wave[sort_array], response[sort_array]

   


def eff_wave(passband, model=None):
    """
    Return the effective wavelength of a photometric passband.
    
    The effective wavelength is defined as the average wavelength weighted with
    the response curve, and is given in angstrom.
    
    >>> eff_wave('2MASS.J')
    12412.136241640892
    
    If you give model fluxes as an extra argument, the wavelengths will take
    these into account to calculate the true effective wavelength (e.g., 
    [VanDerBliek1996]_, Eq 2.)
    
    @param passband: photometric passband
    @type passband: str ('SYSTEM.FILTER') or array/list of str
    @param model: model wavelength and fluxes
    @type model: tuple of 1D arrays (wave,flux)
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
        
        # If the passband is not defined, set it to nan, otherwise compute the
        # effective wavelength.
        try:
            wave, response = get_response(ipassband)
            
            # If a model is given take it into account
            if model is None:
                this_eff_wave = np.average(wave, weights=response)
            else:
                # Interpolate response curve onto higher resolution model and
                # take weighted average
                #is_response = response > 1e-10
                #start_response = wave[is_response].min()
                #end_response = wave[is_response].max()
                fluxm = 10**np.interp(np.log10(wave), np.log10(model[0]),
                                      np.log10(model[1]))
                this_eff_wave = np.trapz(wave*fluxm*response, x=wave) / \
                                np.trapz(fluxm*response, x=wave)
        
        # If the passband is not defined, set the effective wavelength to nan
        except IOError:
            this_eff_wave = np.nan
            
        # Remember the result    
        my_eff_wave.append(this_eff_wave)
    
    # And provide an intuitive return value: if the user gave a single passband,
    # return a single wavelength. Otherwise return an array
    if single_band:
        my_eff_wave = my_eff_wave[0]
    else:
        my_eff_wave = np.array(my_eff_wave, float)
    
    return my_eff_wave


def list_response(name='*'):
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
    responses = sorted(glob.glob(os.path.join(os.path.dirname(__file__),
                                 'ptf', name)))
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
    if os.path.isfile(photfile) and not kwargs['force']:
        raise ValueError, 'bandpass {0} already exists'.format(photfile)
    elif passband in custom_passbands:
        logger.debug('Overwriting previous definition of {0}'.format(passband))
    custom_passbands[passband] = dict(response=(wave, response))
    
    # Set effective wavelength
    kwargs.setdefault('type', 'CCD')
    kwargs.setdefault('eff_wave', eff_wave(passband, det_type=kwargs['type']))
    kwargs.setdefault('transmission', np.trapz(response, x=wave))
    # Add info for zeropoints.dat file: make sure wherever "lit" is part of the
    # name, we replace it with "0". Then, we overwrite any existing information
    # with info given
    myrow = get_info([kwargs['copy_from']])
    for name in myrow.dtype.names:
        if 'lit' in name:
            myrow[name] = 0
        myrow[name] = kwargs.pop(name, myrow[name])
        
    # Remove memoized things to be sure to have the most up-to-date info
    del decorators.memory[__name__]
    
    # Finally add the info:
    custom_passbands[passband]['zp'] = myrow
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
    
    
    

if __name__ == "__main__":
    import doctest
    doctest.testmod()