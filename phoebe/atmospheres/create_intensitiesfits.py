"""
Create a passband FITS file for usage with ``create_atmospherefits.py``.
"""
import numpy as np
try: # Pyfits now integrated in astropy
    import pyfits as pf
except:
    import astropy.io.fits as pf
from phoebe.atmospheres.passbands import *
import glob
import os.path
import phoebe


logger = logging.getLogger("ATM.CREAINTFITS")


def add_mu_table(wave, table, logg, teff, mu, outfile,vturb,logz,reference):
    # Adds table to fits file. Identical for all models giving specific intensities.
    # Only intended to be used by convert_to_fits functions.


    # Prepare fits file if it does not exist
    if not os.path.isfile(outfile):
        hdulist = pf.HDUList([])
        hdulist.append(pf.PrimaryHDU(np.array([[0,0]])))
        hd = hdulist[0].header
        hd.set('vturb', vturb, 'Microturb vel (km/s)')
        hd.set('logz', logz, 'Metallicity wrt Solar')
        hd.set('ref', reference, 'Table reference')
        hdulist.writeto(outfile)



    cols = []
    # sort wavelengths (ascending)
    wave_order = np.argsort(wave)
    wave = wave[wave_order]
    table = table[wave_order,:]

    # sort mu (descending)
    mu_order = np.argsort(mu)
    mu=mu[mu_order[::-1]]
    table = table[:,mu_order[::-1]]

    cols.append(pf.Column(name='wavelength', format='E', array=wave))

    for i in range(len(mu)):
        mu_ = mu[i]
        cols.append(pf.Column(name="%1.06f"%mu_, format='E', array=table[:,i]))

    # Add 0-flux array for mu=0 to facilitate automatic integration of the surface of the star
    cols.append(pf.Column(name="0.000000", format='E', array=table[:,0]*0))

    newtable = pf.new_table(pf.ColDefs(cols))
    newtable.header.set('EXTNAME', "T%05d_logg%01.02f" %(teff,logg), "name of the extension")
    newtable.header.set('teff', teff, 'Effective temperature (K)')
    newtable.header.set('logg', logg, 'Log(g)')
    logger.debug("Writing to: %s",repr(outfile))

    pf.append(outfile, newtable.data,newtable.header)


    # todo: check if headers are also kept



def process_ascii(**kwargs):
    """
    Reads ascii files with specific intensities and converts them into a table.

    Currently supports only castelli-kurucz 2004 format, with one mu-angle per file.

    Name of ascii-inputfile and fits-outputfile should be given as arguments. If vturb, logz (metallicity wrt solar), h (hydrogen fraction) reference
    are provided these will be stored in the headers.
    mu = cos(theta)

    Wavelengths in angstrom.
    Intensities in Flambda. Are converted to Kurucz's units.

    @keyword fileformat: identifier of the file format (dicatates the way the files are parsed)
    @type fileformat: string
    @keyword asciifiles: list of ascii files to be parsed
    @type asciifiles: list of strings
    @keyword outfile_prefix: prefix for fits file
    @type outfile_prefix: string
    @keyword vturb: microturbulent velocity
    @type vturb: float
    @keyword logz: metallicity
    @type logz: float
    @keyword reference: paper or website reference for the table
    @type reference: string

    """
    fileformat = kwargs.get('fileformat','ck2004_ll201505')
    asciifiles = kwargs.get('asciifiles')
    outfile_prefix = kwargs.get('outfile_prefix', "NoName")
    vturb = kwargs.get('vturb', "UNDEFINED")
    reference = kwargs.get('reference', "ck2004_ll201505")

    table = []

    asciifiles.sort()

    if fileformat=='ck2004_ll201505':
        # get all unique combinations of the stellar parameters
        # based on these, we will collect all my angles per stellar atm model
        # and make one fits extension per set

        # file names are of the form:
        # T05250G00M05.M0.001.spc
        unique_metallicities = np.unique([os.path.basename(f)[10:12] for f in asciifiles])

        for metal in unique_metallicities:
            logger.debug("Unique metallicity: %s",repr(metal))
            # we make one file per metallicity
            # select all models with this metallicity:
            unique_mods_1metal = np.unique([os.path.basename(f)[0:12] for f in asciifiles if os.path.basename(f)[10:12]==metal])

            for unique_model in unique_mods_1metal:
                logger.debug("Model: %s",repr(unique_model))
                teff = float(unique_model[1:6])
                logg = float(unique_model[7:9])/10.
                logz_string = unique_model[9:12]
                logz = float(logz_string[1:])/10.
                if unique_model[9]=='M': logz*=-1

                # now grab all the different my angles
                files_1model = [f for f in asciifiles if os.path.basename(f)[0:12]==unique_model]

                # and make one table from these files
                wavls, intens,mus = None,[],[]
                for infile in files_1model:
                    logger.debug(" -- Reading file: %s",repr(infile))

                    mu = float(os.path.basename(infile)[14:19])
                    wavl_array, intens_array = np.loadtxt(infile,unpack=True)
                    if wavls==None:
                        wavls = wavl_array
                    #todo: check if wavls are the same as those in new file!
                    mus.append(mu)
                    intens.append(intens_array)


                # now save to a fits extension
                outfile = '%s_%s.fits'%(outfile_prefix,logz_string)
                add_mu_table(wave=wavls, table=np.array(intens).T, logg=logg, teff=teff, mu=np.array(mus), outfile=outfile,\
                             vturb=vturb,logz=logz,reference=reference)



if __name__ == "__main__":
    phoebe.get_basic_logger(clevel='debug')
    import argparse

    #-- parse files given on command line
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='*.spc', nargs='+',
        help='Filenames (can include the path) of ascii files to be converted. Wildcards such as * can be used.')


    args = parser.parse_args()
    files = args.files #glob.glob('spectra/*.spc')

    outfile_prefix = 'out_fitsfiles/CK2004LL201505'
    process_ascii(outfile_prefix=outfile_prefix, asciifiles=files,vturb=1, reference='CK2004LL201505')