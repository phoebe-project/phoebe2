import os
try: # Pyfits now integrated in astropy
    import pyfits
except:
    import astropy.io.fits as pyfits

import numpy as np

def write_array(arr,filename,names=(),units=(),header_dict={},ext='new',close=True):
    """
    Write or add an array to a FITS file.
    
    If 'filename' refers to an existing file, the list of arrays will be added
    (ext='new') to the HDUlist or replace an existing HDU (ext=integer). Else,
    a new file will be created.
    
    Names and units should be given as a list of strings, in the same order as
    the list of arrays.
    
    A header_dictionary can be given, it is used to update an existing header
    or create a new one if the extension is new.
    
    Instead of writing the file, you can give a hdulist and append to it.
    Supply a HDUList for 'filename', and set close=False
    """
    if isinstance(filename,str) and not os.path.isfile(filename):
            primary = np.array([[0]])
            hdulist = pyfits.HDUList([pyfits.PrimaryHDU(primary)])
            hdulist.writeto(filename)
            
    if isinstance(filename,str):
        hdulist = pyfits.open(filename,mode='update')
    else:
        hdulist = filename
    
    #-- create the table HDU
    cols = []
    for i,name in enumerate(names):
        format = arr[i].dtype.str.lower().replace('|','').replace('s','a').replace('>','')
        format = format.replace('b1','L').replace('<','')
        if format=='f8':
            format = 'D'
        if isinstance(units,dict):
            unit = name in units and units[name] or 'NA'
        elif len(units)>i:
            unit = units[i]
        else:
            unit = 'NA'
        cols.append(pyfits.Column(name=name,format=format,array=arr[i],unit=unit))
    tbhdu = pyfits.new_table(pyfits.ColDefs(cols))
    
    #   put it in the right place
    if ext=='new' or ext==len(hdulist):
        hdulist.append(tbhdu)
        ext = -1
    else:
        hdulist[ext] = tbhdu
    
    #-- take care of the header:
    if len(header_dict):
        for key in header_dict:
            hdulist[ext].header.update(key,header_dict[key])
    
    if close:
        hdulist.close()
    else:
        return hdulist

#}