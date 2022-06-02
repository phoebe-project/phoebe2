#from phoebe.c import h, c, k_B
#from phoebe import u
from phoebe import __version__ as phoebe_version
from phoebe import conf, mpi
from phoebe.utils import _bytes

# NOTE: we'll import directly from astropy here to avoid
# circular imports BUT any changes to these units/constants
# inside phoebe will be ignored within passbands
from astropy.constants import h, c, k_B, sigma_sb
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

import numpy as np
from scipy import interpolate, integrate
from scipy.optimize import curve_fit as cfit
from datetime import datetime
import marshal
import pickle
import types
import libphoebe
import os
import sys
import glob
import shutil
import json
import time

# NOTE: python3 only
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError, HTTPError

from phoebe.utils import parse_json

import logging
logger = logging.getLogger("PASSBANDS")
logger.addHandler(logging.NullHandler())

# define the URL to query for online passbands.  See tables.phoebe-project.org
# repo for the source-code of the server
_url_tables_server = 'http://tables.phoebe-project.org'
# comment out the following line if testing tables.phoebe-project.org server locally:
# _url_tables_server = 'http://localhost:5555'

# Future atmosphere tables could exist in the passband files, but the current
# release won't be able to handle those.
_supported_atms = ['blackbody', 'ck2004', 'phoenix', 'extern_atmx', 'extern_planckint']

# Global passband table. This dict should never be tinkered with outside
# of the functions in this module; it might be nice to make it read-only
# at some point.
_pbtable = {}

_initialized = False
_online_passbands = {}
_online_passband_failedtries = 0

_pbdir_global = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/passbands'))+'/'

# if we're in a virtual environment then we want don't want to use the home directory
# this check may fail for Python 3
if hasattr(sys, 'real_prefix'):
    # then we're running in a virtualenv
    _pbdir_local = os.path.join(sys.prefix, '.phoebe/atmospheres/tables/passbands/')
else:
    _pbdir_local = os.path.abspath(os.path.expanduser('~/.phoebe/atmospheres/tables/passbands'))+'/'

if not os.path.exists(_pbdir_local):
    if not mpi.within_mpirun or mpi.myrank == 0:
        logger.info("creating directory {}".format(_pbdir_local))
        os.makedirs(_pbdir_local)

_pbdir_env = os.getenv('PHOEBE_PBDIR', None)


_pbdir_env = os.getenv('PHOEBE_PBDIR', None)

def _dict_without_keys(d, skip_keys=[]):
    return {k:v for k,v in d.items() if k not in skip_keys}

class Passband:
    def __init__(self, ptf=None, pbset='Johnson', pbname='V', effwl=5500.0,
                 wlunits=u.AA, calibrated=False, reference='', version=1.0,
                 comments='', oversampling=1, ptf_order=3, from_file=False):
        """
        <phoebe.atmospheres.passbands.Passband> class holds data and tools for
        passband-related computations, such as blackbody intensity, model
        atmosphere intensity, etc.

        Step #1: initialize passband object

        ```py
        pb = Passband(ptf='JOHNSON.V', pbset='Johnson', pbname='V', effwl=5500.0, wlunits=u.AA, calibrated=True, reference='ADPS', version=1.0, comments='')
        ```

        Step #2: compute intensities for blackbody radiation:

        ```py
        pb.compute_blackbody_response()
        ```

        Step #3: compute Castelli & Kurucz (2004) intensities. To do this,
        the tables/ck2004 directory needs to be populated with non-filtered
        intensities available for download from %static%/ck2004.tar.

        ```py
        atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/ck2004'))
        pb.compute_ck2004_response(atmdir)
        ```

        Step #4: -- optional -- import WD tables for comparison. This can only
        be done if the passband is in the list of supported passbands in WD.
        The WD index of the passband is passed to the import_wd_atmcof()
        function below as the last argument.

        ```py
        from phoebe.atmospheres import atmcof
        atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/wd'))
        atmcof.init(atmdir+'/atmcofplanck.dat', atmdir+'/atmcof.dat')
        pb.import_wd_atmcof(atmdir+'/atmcofplanck.dat', atmdir+'/atmcof.dat', 7)
        ```

        Step #5: save the passband file:

        ```py
        atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/passbands'))
        pb.save(atmdir + '/johnson_v.ptf')
        ```

        From now on you can use `pbset`:`pbname` as a passband qualifier, i.e.
        Johnson:V for the example above. Further details on supported model
        atmospheres are available by issuing:

        ```py
        pb.content
        ```

        see <phoebe.atmospheres.passbands.content>

        Arguments
        ----------
        * `ptf` (string, optional, default=None): passband transmission file: a
            2-column file with wavelength in @wlunits and transmission in
            arbitrary units.
        * `pbset` (string, optional, default='Johnson'): name of the passband
            set (i.e. Johnson).
        * `pbname` (string, optional, default='V'): name of the passband name
            (i.e. V).
        * `effwl` (float, optional, default=5500.0): effective wavelength in
            `wlunits`.
        * `wlunits` (unit, optional, default=u.AA): wavelength units from
            astropy.units used in `ptf` and `effwl`.
        * `calibrated` (bool, optional, default=False): true if transmission is
            in true fractional light, false if it is in relative proportions.
        * `reference` (string, optional, default=''): passband transmission data
            reference (i.e. ADPS).
        * `version` (float, optional, default=1.0): file version.
        * `comments` (string, optional, default=''): any additional comments
            about the passband.
        * `oversampling` (int, optional, default=1): the multiplicative factor
            of PTF dispersion to attain higher integration accuracy.
        * `ptf_order` (int, optional, default=3): spline order for fitting
            the passband transmission function.
        * `from_file` (bool, optional, default=False): a switch that instructs
            the class instance to skip all calculations and load all data from
            the file passed to the <phoebe.atmospheres.passbands.Passband.load>
            method.

        Returns
        ---------
        * an instatiated <phoebe.atmospheres.passbands.Passband> object.
        """
        if "'" in pbset or '"' in pbset:
            raise ValueError("pbset cannot contain quotation marks")
        if "'" in pbname or '"' in pbname:
            raise ValueError("pbset cannot contain quotation marks")

        self.h = h.value
        self.c = c.value
        self.k = k_B.value

        if from_file:
            return

        # Initialize content list; each method that adds any content
        # to the passband file needs to add a corresponding label to the
        # content list.
        self.content = []

        # Basic passband properties:
        self.pbset = pbset
        self.pbname = pbname
        self.effwl = effwl
        self.calibrated = calibrated
        self.reference = reference
        self.version = version
        self.comments = comments

        # Initialize an empty timestamp. This will get set by calling the save() method.
        self.timestamp = None

        # Passband transmission function table:
        ptf_table = np.loadtxt(ptf).T
        ptf_table[0] = ptf_table[0]*wlunits.to(u.m)
        self.ptf_table = {'wl': np.array(ptf_table[0]), 'fl': np.array(ptf_table[1])}

        # Working (optionally oversampled) wavelength array:
        self.wl = np.linspace(self.ptf_table['wl'][0], self.ptf_table['wl'][-1], oversampling*len(self.ptf_table['wl']))

        # Spline fit to the energy-weighted passband transmission function table:
        self.ptf_order = ptf_order
        self.ptf_func = interpolate.splrep(self.ptf_table['wl'], self.ptf_table['fl'], s=0, k=ptf_order)
        self.ptf = lambda wl: interpolate.splev(wl, self.ptf_func)
        self.ptf_area = interpolate.splint(self.wl[0], self.wl[-1], self.ptf_func, 0)

        # Spline fit to the photon-weighted passband transmission function table:
        self.ptf_photon_func = interpolate.splrep(self.ptf_table['wl'], self.ptf_table['fl']*self.ptf_table['wl'], s=0, k=ptf_order)
        self.ptf_photon = lambda wl: interpolate.splev(wl, self.ptf_photon_func)
        self.ptf_photon_area = interpolate.splint(self.wl[0], self.wl[-1], self.ptf_photon_func, 0)

        # Initialize (empty) history:
        self.history = {}

    def __repr__(self):
        return '<Passband: %s:%s>' % (self.pbset, self.pbname)

    def __str__(self):
        # old passband files do not have versions embedded, that is why we have to do this:
        if not hasattr(self, 'version') or self.version is None:
            self.version = 1.0
        return('Passband: %s:%s\nVersion:  %1.1f\nProvides: %s' % (self.pbset, self.pbname, self.version, self.content))

    def on_updated_ptf(self, ptf, wlunits=u.AA, oversampling=1, ptf_order=3):
        """
        When passband transmission function is updated, this function updates
        all related meta-fields in the passband structure. It does *not* update
        any tables, only the header information.
        """

        ptf_table = np.loadtxt(ptf).T
        ptf_table[0] = ptf_table[0]*wlunits.to(u.m)
        self.ptf_table = {'wl': np.array(ptf_table[0]), 'fl': np.array(ptf_table[1])}

        self.wl = np.linspace(self.ptf_table['wl'][0], self.ptf_table['wl'][-1], oversampling*len(self.ptf_table['wl']))

        self.ptf_order = ptf_order
        self.ptf_func = interpolate.splrep(self.ptf_table['wl'], self.ptf_table['fl'], s=0, k=ptf_order)
        self.ptf = lambda wl: interpolate.splev(wl, self.ptf_func)
        self.ptf_area = interpolate.splint(self.wl[0], self.wl[-1], self.ptf_func, 0)

        # Spline fit to the photon-weighted passband transmission function table:
        self.ptf_photon_func = interpolate.splrep(self.ptf_table['wl'], self.ptf_table['fl']*self.ptf_table['wl'], s=0, k=ptf_order)
        self.ptf_photon = lambda wl: interpolate.splev(wl, self.ptf_photon_func)
        self.ptf_photon_area = interpolate.splint(self.wl[0], self.wl[-1], self.ptf_photon_func, 0)

        return

    def save(self, archive, overwrite=True, update_timestamp=True, history_entry=''):
        """
        Saves the passband file in the fits format.

        Arguments
        ----------
        * `archive` (string): filename of the saved file
        * `overwrite` (bool, optional, default=True): whether to overwrite an
            existing file with the same filename as provided in `archive`
        * `update_timestamp` (bool, optional, default=True): whether to update
            the stored timestamp with the current time.
        * `history_entry` (string, optional): history entry to append to the
            fits file.  Note that previous entries will be maintained if
            (and only if) overwriting an existing file with `overwrite=True`.
        """

        # Timestamp is used for passband versioning.
        timestamp = time.ctime() if update_timestamp else self.timestamp

        header = fits.Header()
        header['PHOEBEVN'] = phoebe_version
        header['TIMESTMP'] = timestamp
        header['PBSET'] = self.pbset
        header['PBNAME'] = self.pbname
        header['EFFWL'] = self.effwl
        header['CALIBRTD'] = self.calibrated
        header['VERSION'] = self.version
        header['COMMENTS'] = self.comments
        header['REFERENC'] = self.reference
        header['PTFORDER'] = self.ptf_order
        header['PTFEAREA'] = self.ptf_area
        header['PTFPAREA'] = self.ptf_photon_area

        header['CONTENT'] = str(self.content)

        # Add all existing history entries:
        for h in self.history.keys():
            header['HISTORY'] = h + ': ' + self.history[h] + '-END-'

        # Append any new history entry:
        if history_entry:
            header['HISTORY'] = '%s: %s' % (timestamp, history_entry) + '-END-'
            self.history[timestamp] = history_entry

        if 'extern_planckint:Inorm' in self.content or 'extern_atmx:Inorm' in self.content:
            header['WD_IDX'] = self.extern_wd_idx

        data = []

        # Header:
        primary_hdu = fits.PrimaryHDU(header=header)
        data.append(primary_hdu)

        # Tables:
        ptf_table = Table(self.ptf_table)
        data.append(fits.table_to_hdu(Table(self.ptf_table, meta={'extname': 'PTFTABLE'})))

        if 'blackbody:Inorm' in self.content:
            bb_func = Table({'teff': self._bb_func_energy[0], 'logi_e': self._bb_func_energy[1], 'logi_p': self._bb_func_photon[1]}, meta={'extname': 'BB_FUNC'})
            data.append(fits.table_to_hdu(bb_func))

        if 'blackbody:ext' in self.content:
            data.append(fits.table_to_hdu(Table({'teff': self._bb_extinct_axes[0]}, meta={'extname': 'BB_TEFFS'})))
            data.append(fits.table_to_hdu(Table({'ebv': self._bb_extinct_axes[1]}, meta={'extname': 'BB_EBVS'})))
            data.append(fits.table_to_hdu(Table({'rv': self._bb_extinct_axes[2]}, meta={'extname': 'BB_RVS'})))

        if 'ck2004:Inorm' in self.content:
            ck_teffs, ck_loggs, ck_abuns = self._ck2004_axes
            data.append(fits.table_to_hdu(Table({'teff': ck_teffs}, meta={'extname': 'CK_TEFFS'})))
            data.append(fits.table_to_hdu(Table({'logg': ck_loggs}, meta={'extname': 'CK_LOGGS'})))
            data.append(fits.table_to_hdu(Table({'abun': ck_abuns}, meta={'extname': 'CK_ABUNS'})))

        if 'ck2004:Imu' in self.content:
            ck_mus = self._ck2004_intensity_axes[-1]
            data.append(fits.table_to_hdu(Table({'mu': ck_mus}, meta={'extname': 'CK_MUS'})))

        if 'ck2004:ext' in self.content:
            ck_ebvs = self._ck2004_extinct_axes[-2]
            ck_rvs = self._ck2004_extinct_axes[-1]
            data.append(fits.table_to_hdu(Table({'ebv': ck_ebvs}, meta={'extname': 'CK_EBVS'})))
            data.append(fits.table_to_hdu(Table({'rv': ck_rvs}, meta={'extname': 'CK_RVS'})))

        if 'phoenix:Inorm' in self.content:
            ph_teffs, ph_loggs, ph_abuns = self._phoenix_axes
            data.append(fits.table_to_hdu(Table({'teff': ph_teffs}, meta={'extname': 'PH_TEFFS'})))
            data.append(fits.table_to_hdu(Table({'logg': ph_loggs}, meta={'extname': 'PH_LOGGS'})))
            data.append(fits.table_to_hdu(Table({'abun': ph_abuns}, meta={'extname': 'PH_ABUNS'})))

        if 'phoenix:Imu' in self.content:
            ph_mus = self._phoenix_intensity_axes[-1]
            data.append(fits.table_to_hdu(Table({'mu': ph_mus}, meta={'extname': 'PH_MUS'})))

        if 'phoenix:ext' in self.content:
            ph_ebvs = self._phoenix_extinct_axes[-2]
            ph_rvs = self._phoenix_extinct_axes[-1]
            data.append(fits.table_to_hdu(Table({'ebv': ph_ebvs}, meta={'extname': 'PH_EBVS'})))
            data.append(fits.table_to_hdu(Table({'rv': ph_rvs}, meta={'extname': 'PH_RVS'})))

        # Data:
        if 'blackbody:ext' in self.content:
            data.append(fits.ImageHDU(self._bb_extinct_energy_grid, name='BBEGRID'))
            data.append(fits.ImageHDU(self._bb_extinct_photon_grid, name='BBPGRID'))

        if 'ck2004:Inorm' in self.content:
            data.append(fits.ImageHDU(self._ck2004_energy_grid, name='CKNEGRID'))
            data.append(fits.ImageHDU(self._ck2004_photon_grid, name='CKNPGRID'))

        if 'ck2004:Imu' in self.content:
            data.append(fits.ImageHDU(self._ck2004_Imu_energy_grid, name='CKFEGRID'))
            data.append(fits.ImageHDU(self._ck2004_Imu_photon_grid, name='CKFPGRID'))

        if 'ck2004:ld' in self.content:
            data.append(fits.ImageHDU(self._ck2004_ld_energy_grid, name='CKLEGRID'))
            data.append(fits.ImageHDU(self._ck2004_ld_photon_grid, name='CKLPGRID'))

        if 'ck2004:ldint' in self.content:
            data.append(fits.ImageHDU(self._ck2004_ldint_energy_grid, name='CKIEGRID'))
            data.append(fits.ImageHDU(self._ck2004_ldint_photon_grid, name='CKIPGRID'))

        if 'ck2004:ext' in self.content:
            data.append(fits.ImageHDU(self._ck2004_extinct_energy_grid, name='CKXEGRID'))
            data.append(fits.ImageHDU(self._ck2004_extinct_photon_grid, name='CKXPGRID'))

        if 'phoenix:Inorm' in self.content:
            data.append(fits.ImageHDU(self._phoenix_energy_grid, name='PHNEGRID'))
            data.append(fits.ImageHDU(self._phoenix_photon_grid, name='PHNPGRID'))

        if 'phoenix:Imu' in self.content:
            data.append(fits.ImageHDU(self._phoenix_Imu_energy_grid, name='PHFEGRID'))
            data.append(fits.ImageHDU(self._phoenix_Imu_photon_grid, name='PHFPGRID'))

        if 'phoenix:ld' in self.content:
            data.append(fits.ImageHDU(self._phoenix_ld_energy_grid, name='PHLEGRID'))
            data.append(fits.ImageHDU(self._phoenix_ld_photon_grid, name='PHLPGRID'))

        if 'phoenix:ldint' in self.content:
            data.append(fits.ImageHDU(self._phoenix_ldint_energy_grid, name='PHIEGRID'))
            data.append(fits.ImageHDU(self._phoenix_ldint_photon_grid, name='PHIPGRID'))

        if 'phoenix:ext' in self.content:
            data.append(fits.ImageHDU(self._phoenix_extinct_energy_grid, name='PHXEGRID'))
            data.append(fits.ImageHDU(self._phoenix_extinct_photon_grid, name='PHXPGRID'))

        pb = fits.HDUList(data)
        pb.writeto(archive, overwrite=overwrite)

    @classmethod
    def load(cls, archive, load_content=True):
        """
        Loads the passband contents from a fits file.

        Arguments
        ----------
        * `archive` (string): filename of the passband (in FITS format)
        * `load_content` (bool, optional, default=True): whether to load all
            table contents.  If False, only the headers will be loaded into
            the structure.

        Returns
        --------
        * the instantiated <phoebe.atmospheres.passbands.Passband> object.
        """

        logger.debug("loading passband from {}".format(archive))

        self = cls(from_file=True)
        with fits.open(archive) as hdul:
            header = hdul['primary'].header

            self.phoebe_version = header['phoebevn']
            self.version = header['version']
            self.timestamp = header['timestmp']

            self.pbset = header['pbset']
            self.pbname = header['pbname']
            self.effwl = header['effwl']
            self.calibrated = header['calibrtd']
            self.comments = header['comments']
            self.reference = header['referenc']
            self.ptf_order = header['ptforder']
            self.ptf_area = header['ptfearea']
            self.ptf_photon_area = header['ptfparea']

            self.content = eval(header['content'], {'__builtins__':None}, {})

            try:
                history = ''.join(header['HISTORY']).split('-END-')
            except KeyError:
                history = []

            self.history = {h.split(': ')[0]: ': '.join(h.split(': ')[1:]) for h in history if len(h.split(': ')) > 1}

            self.ptf_table = hdul['ptftable'].data

            # Rebuild ptf() and photon_ptf() functions:
            self.ptf_func = interpolate.splrep(self.ptf_table['wl'], self.ptf_table['fl'], s=0, k=self.ptf_order)
            self.ptf = lambda wl: interpolate.splev(wl, self.ptf_func)
            self.ptf_photon_func = interpolate.splrep(self.ptf_table['wl'], self.ptf_table['fl']*self.ptf_table['wl'], s=0, k=self.ptf_order)
            self.ptf_photon = lambda wl: interpolate.splev(wl, self.ptf_photon_func)

            if load_content:
                if 'extern_planckint:Inorm' in self.content or 'extern_atmx:Inorm' in self.content:
                    atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/wd'))
                    planck = os.path.join(atmdir+'/atmcofplanck.dat').encode('utf8')
                    atm = os.path.join(atmdir+'/atmcof.dat').encode('utf8')

                    self.wd_data = libphoebe.wd_readdata(planck, atm)
                    self.extern_wd_idx = header['wd_idx']

                if 'blackbody:Inorm' in self.content:
                    self._bb_func_energy = (hdul['bb_func'].data['teff'], hdul['bb_func'].data['logi_e'], 3)
                    self._bb_func_photon = (hdul['bb_func'].data['teff'], hdul['bb_func'].data['logi_p'], 3)
                    self._log10_Inorm_bb_energy = lambda Teff: interpolate.splev(Teff, self._bb_func_energy)
                    self._log10_Inorm_bb_photon = lambda Teff: interpolate.splev(Teff, self._bb_func_photon)

                if 'blackbody:ext' in self.content:
                    self._bb_extinct_axes = (np.array(list(hdul['bb_teffs'].data['teff'])), np.array(list(hdul['bb_ebvs'].data['ebv'])), np.array(list(hdul['bb_rvs'].data['rv'])))
                    self._bb_extinct_energy_grid = hdul['bbegrid'].data
                    self._bb_extinct_photon_grid = hdul['bbpgrid'].data

                if 'ck2004:Inorm' in self.content:
                    self._ck2004_axes = (np.array(list(hdul['ck_teffs'].data['teff'])), np.array(list(hdul['ck_loggs'].data['logg'])), np.array(list(hdul['ck_abuns'].data['abun'])))
                    self._ck2004_energy_grid = hdul['cknegrid'].data
                    self._ck2004_photon_grid = hdul['cknpgrid'].data

                if 'ck2004:Imu' in self.content:
                    self._ck2004_intensity_axes = (np.array(list(hdul['ck_teffs'].data['teff'])), np.array(list(hdul['ck_loggs'].data['logg'])), np.array(list(hdul['ck_abuns'].data['abun'])), np.array(list(hdul['ck_mus'].data['mu'])))
                    self._ck2004_Imu_energy_grid = hdul['ckfegrid'].data
                    self._ck2004_Imu_photon_grid = hdul['ckfpgrid'].data

                if 'ck2004:ld' in self.content:
                    self._ck2004_ld_energy_grid = hdul['cklegrid'].data
                    self._ck2004_ld_photon_grid = hdul['cklpgrid'].data

                if 'ck2004:ldint' in self.content:
                    self._ck2004_ldint_energy_grid = hdul['ckiegrid'].data
                    self._ck2004_ldint_photon_grid = hdul['ckipgrid'].data

                if 'ck2004:ext' in self.content:
                    self._ck2004_extinct_axes = (np.array(list(hdul['ck_teffs'].data['teff'])), np.array(list(hdul['ck_loggs'].data['logg'])), np.array(list(hdul['ck_abuns'].data['abun'])), np.array(list(hdul['ck_ebvs'].data['ebv'])), np.array(list(hdul['ck_rvs'].data['rv'])))
                    self._ck2004_extinct_energy_grid = hdul['ckxegrid'].data
                    self._ck2004_extinct_photon_grid = hdul['ckxpgrid'].data

                if 'phoenix:Inorm' in self.content:
                    self._phoenix_axes = (np.array(list(hdul['ph_teffs'].data['teff'])), np.array(list(hdul['ph_loggs'].data['logg'])), np.array(list(hdul['ph_abuns'].data['abun'])))
                    self._phoenix_energy_grid = hdul['phnegrid'].data
                    self._phoenix_photon_grid = hdul['phnpgrid'].data

                if 'phoenix:Imu' in self.content:
                    self._phoenix_intensity_axes = (np.array(list(hdul['ph_teffs'].data['teff'])), np.array(list(hdul['ph_loggs'].data['logg'])), np.array(list(hdul['ph_abuns'].data['abun'])), np.array(list(hdul['ph_mus'].data['mu'])))
                    self._phoenix_Imu_energy_grid = hdul['phfegrid'].data
                    self._phoenix_Imu_photon_grid = hdul['phfpgrid'].data

                if 'phoenix:ld' in self.content:
                    self._phoenix_ld_energy_grid = hdul['phlegrid'].data
                    self._phoenix_ld_photon_grid = hdul['phlpgrid'].data

                if 'phoenix:ldint' in self.content:
                    self._phoenix_ldint_energy_grid = hdul['phiegrid'].data
                    self._phoenix_ldint_photon_grid = hdul['phipgrid'].data

                if 'phoenix:ext' in self.content:
                    self._phoenix_extinct_axes = (np.array(list(hdul['ph_teffs'].data['teff'])),np.array(list(hdul['ph_loggs'].data['logg'])), np.array(list(hdul['ph_abuns'].data['abun'])), np.array(list(hdul['ph_ebvs'].data['ebv'])), np.array(list(hdul['ph_rvs'].data['rv'])))
                    self._phoenix_extinct_energy_grid = hdul['phxegrid'].data
                    self._phoenix_extinct_photon_grid = hdul['phxpgrid'].data

        return self

    def _planck(self, lam, Teff):
        """
        Computes monochromatic blackbody intensity in W/m^3 using the
        Planck function.

        Arguments
        -----------
        * `lam` (float/array): wavelength in m
        * `Teff` (float/array): effective temperature in K

        Returns
        --------
        * monochromatic blackbody intensity
        """

        return 2*self.h*self.c*self.c/lam**5 * 1./(np.exp(self.h*self.c/lam/self.k/Teff)-1)

    def _planck_deriv(self, lam, Teff):
        """
        Computes the derivative of the monochromatic blackbody intensity using
        the Planck function.

        Arguments
        -----------
        * `lam` (float/array): wavelength in m
        * `Teff` (float/array): effective temperature in K

        Returns
        --------
        * the derivative of monochromatic blackbody intensity
        """

        expterm = np.exp(self.h*self.c/lam/self.k/Teff)
        return 2*self.h*self.c*self.c/self.k/Teff/lam**7 * (expterm-1)**-2 * (self.h*self.c*expterm-5*lam*self.k*Teff*(expterm-1))

    def _planck_spi(self, lam, Teff):
        """
        Computes the spectral index of the monochromatic blackbody intensity
        using the Planck function. The spectral index is defined as:

            B(lambda) = 5 + d(log I)/d(log lambda),

        where I is the Planck function.

        Arguments
        -----------
        * `lam` (float/array): wavelength in m
        * `Teff` (float/array): effective temperature in K

        Returns
        --------
        * the spectral index of monochromatic blackbody intensity
        """

        hclkt = self.h*self.c/lam/self.k/Teff
        expterm = np.exp(hclkt)
        return hclkt * expterm/(expterm-1)

    def _bb_intensity(self, Teff, photon_weighted=False):
        """
        Computes mean passband intensity using blackbody atmosphere:

        I_pb^E = \int_\lambda I(\lambda) P(\lambda) d\lambda / \int_\lambda P(\lambda) d\lambda
        I_pb^P = \int_\lambda \lambda I(\lambda) P(\lambda) d\lambda / \int_\lambda \lambda P(\lambda) d\lambda

        Superscripts E and P stand for energy and photon, respectively.

        Arguments
        -----------
        * `Teff` (float/array): effective temperature in K
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        ------------
        * mean passband intensity using blackbody atmosphere.
        """

        if photon_weighted:
            pb = lambda w: w*self._planck(w, Teff)*self.ptf(w)
            return integrate.quad(pb, self.wl[0], self.wl[-1])[0]/self.ptf_photon_area
        else:
            pb = lambda w: self._planck(w, Teff)*self.ptf(w)
            return integrate.quad(pb, self.wl[0], self.wl[-1])[0]/self.ptf_area

    def _bindex_blackbody(self, Teff, photon_weighted=False):
        """
        Computes the mean boosting index using blackbody atmosphere:

        B_pb^E = \int_\lambda I(\lambda) P(\lambda) B(\lambda) d\lambda / \int_\lambda I(\lambda) P(\lambda) d\lambda
        B_pb^P = \int_\lambda \lambda I(\lambda) P(\lambda) B(\lambda) d\lambda / \int_\lambda \lambda I(\lambda) P(\lambda) d\lambda

        Superscripts E and P stand for energy and photon, respectively.

        Arguments
        ----------
        * `Teff` (float/array): effective temperature in K
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        ------------
        * mean boosting index using blackbody atmosphere.
        """

        if photon_weighted:
            num   = lambda w: w*self._planck(w, Teff)*self.ptf(w)*self._planck_spi(w, Teff)
            denom = lambda w: w*self._planck(w, Teff)*self.ptf(w)
            return integrate.quad(num, self.wl[0], self.wl[-1], epsabs=1e10, epsrel=1e-8)[0]/integrate.quad(denom, self.wl[0], self.wl[-1], epsabs=1e10, epsrel=1e-6)[0]
        else:
            num   = lambda w: self._planck(w, Teff)*self.ptf(w)*self._planck_spi(w, Teff)
            denom = lambda w: self._planck(w, Teff)*self.ptf(w)
            return integrate.quad(num, self.wl[0], self.wl[-1], epsabs=1e10, epsrel=1e-8)[0]/integrate.quad(denom, self.wl[0], self.wl[-1], epsabs=1e10, epsrel=1e-6)[0]

    def compute_blackbody_response(self, Teffs=None):
        """
        Computes blackbody intensities across the entire range of
        effective temperatures. It does this for two regimes, energy-weighted
        and photon-weighted. It then fits a cubic spline to the log(I)-Teff
        values and exports the interpolation functions _log10_Inorm_bb_energy
        and _log10_Inorm_bb_photon.

        Arguments
        ----------
        * `Teffs` (array, optional, default=None): an array of effective
            temperatures. If None, a default array from ~300K to ~500000K with
            97 steps is used. The default array is uniform in log10 scale.
        """

        if Teffs is None:
            log10Teffs = np.linspace(2.5, 5.7, 97) # this corresponds to the 316K-501187K range.
            Teffs = 10**log10Teffs

        # Energy-weighted intensities:
        log10ints_energy = np.array([np.log10(self._bb_intensity(Teff, photon_weighted=False)) for Teff in Teffs])
        self._bb_func_energy = interpolate.splrep(Teffs, log10ints_energy, s=0)
        self._log10_Inorm_bb_energy = lambda Teff: interpolate.splev(Teff, self._bb_func_energy)

        # Photon-weighted intensities:
        log10ints_photon = np.array([np.log10(self._bb_intensity(Teff, photon_weighted=True)) for Teff in Teffs])
        self._bb_func_photon = interpolate.splrep(Teffs, log10ints_photon, s=0)
        self._log10_Inorm_bb_photon = lambda Teff: interpolate.splev(Teff, self._bb_func_photon)

        if 'blackbody:Inorm' not in self.content:
            self.content.append('blackbody:Inorm')

    def impute_atmosphere_grid(self, grid):
        """
        This function imputes the passed atmosphere grid by linear N-D interpolation.
        As grid is passed by reference, it is not necessary to re-assign the table to
        the return value of this function; the return value is provided for convenience
        only, but the grid is changed in place.
        """

        valid_mask = ~np.isnan(grid[...,0])
        coords = np.array(np.nonzero(valid_mask)).T
        values = grid[valid_mask][:,0]
        it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
        filled = it(list(np.ndindex(grid[...,0].shape))).reshape(grid[...,0].shape)
        filled[filled==0] = np.nan
        grid[...,0] = filled
        return grid

    def compute_bb_reddening(self, Teffs=None, Ebv=None, Rv=None, verbose=False):
        """
        Computes mean effect of reddening (a weighted average) on passband using
        blackbody atmosphere and CCM89 prescription of extinction.

        See also:
        * <phoebe.atmospheres.passbands.Passband.compute_ck2004_reddening>
        * <phoebe.atmospheres.passbands.Passband.compute_phoenix_reddening>

        Arguments
        -----------
        * `Teffs` (array or None, optional, default=None): an array of effective
            temperatures. If None, a default array from ~300K to ~500000K with
            97 steps is used. The default array is uniform in log10 scale.
        * `Ebv` (float or None, optional, default=None): color discrepancies E(B-V)
        * `Rv` (float or None, optional, default=None): Extinction factor
            (defined at Av / E(B-V) where Av is the visual extinction in magnitudes)
        * `verbose` (bool, optional, default=False): switch to determine whether
            computing progress should be printed on screen
        """

        if Teffs is None:
            log10Teffs = np.linspace(2.5, 5.7, 97) # this corresponds to the 316K-501187K range.
            Teffs = 10**log10Teffs

        if Ebv is None:
            Ebv = np.linspace(0.,3.,30)

        if Rv is None:
            Rv = np.linspace(2.,6.,16)

        #Make it so that Teffs and Ebv step through a la the CK2004 models
        NTeffs = len(Teffs)
        NEbv = len(Ebv)
        NRv = len(Rv)
        combos = NTeffs*NEbv*NRv
        Teffs = np.repeat(Teffs, int(combos/NTeffs))
        Ebv = np.tile(np.repeat(Ebv, NRv), NTeffs)
        Rv = np.tile(Rv, int(combos/NRv))

        extinctE, extinctP = np.empty(combos), np.empty(combos)

        if verbose:
            print('Computing blackbody reddening corrections for %s:%s.' % (self.pbset, self.pbname))

        # a = libphoebe.CCM89_extinction(self.wl)
        a = libphoebe.gordon_extinction(self.wl)

        for j in range(0,combos):

            pbE = self.ptf(self.wl)*libphoebe.planck_function(self.wl, Teffs[j])
            pbP = self.wl*pbE

            flux_frac = np.exp(-0.9210340371976184*np.dot(a, [Ebv[j]*Rv[j], Ebv[j]]))
            extinctE[j], extinctP[j] = np.dot([pbE/pbE.sum(), pbP/pbP.sum()], flux_frac)

            if verbose:
                sys.stdout.write('\r' + '%0.0f%% done.' % (100*j/(combos-1)))
                sys.stdout.flush()

        if verbose:
            print('')

        self._bb_extinct_axes = (np.unique(Teffs), np.unique(Ebv), np.unique(Rv))

        self._bb_extinct_photon_grid = np.nan*np.ones((len(self._bb_extinct_axes[0]), len(self._bb_extinct_axes[1]), len(self._bb_extinct_axes[2]), 1))
        self._bb_extinct_energy_grid = np.copy(self._bb_extinct_photon_grid)

        for i in range(combos):
            t=(Teffs[i] == self._bb_extinct_axes[0], Ebv[i] == self._bb_extinct_axes[1], Rv[i] == self._bb_extinct_axes[2], 0)
            self._bb_extinct_energy_grid[t] = extinctE[i]
            self._bb_extinct_photon_grid[t] = extinctP[i]

        if 'blackbody:ext' not in self.content:
            self.content.append('blackbody:ext')

    def compute_ck2004_reddening(self, path, Ebv=None, Rv=None, verbose=False):
        """
        Computes mean effect of reddening (a weighted average) on passband using
        ck2004 atmospheres and CCM89 prescription of extinction.

        See also:
        * <phoebe.atmospheres.passbands.Passband.compute_bb_reddening>
        * <phoebe.atmospheres.passbands.Passband.compute_phoenix_reddening>

        Arguments
        ------------
        * `path` (string): path to the directory containing ck2004 SEDs
        * `Ebv` (float or None, optional, default=None): colour discrepancies E(B-V)
        * `Rv` (float or None, optional, default=None): Extinction factor
            (defined at Av / E(B-V) where Av is the visual extinction in magnitudes)
        * `verbose` (bool, optional, default=False): switch to determine whether
            computing progress should be printed on screen
        """

        if Ebv is None:
            Ebv = np.linspace(0.,3.,30)

        if Rv is None:
            Rv = np.linspace(2.,6.,16)

        models = glob.glob(path+'/*fits')
        Nmodels = len(models)

        NEbv = len(Ebv)
        NRv = len(Rv)

        Ns = NEbv*NRv
        combos = Nmodels*Ns

        Ebv1 = np.tile(np.repeat(Ebv, NRv), Nmodels)
        Rv1 = np.tile(Rv, int(combos/NRv))

        # auxilary matrix for storing Ebv and Rv per model
        M = np.rollaxis(np.array([np.split(Ebv1*Rv1, Nmodels), np.split(Ebv1, Nmodels)]), 1)
        M = np.ascontiguousarray(M)

        # Store the length of the filename extensions for parsing:
        offset = len(models[0])-models[0].rfind('.')

        Teff, logg, abun = np.empty(Nmodels), np.empty(Nmodels), np.empty(Nmodels)

        # extinctE , extinctP per model
        extinctE , extinctP = np.empty((Nmodels, Ns)), np.empty((Nmodels, Ns))

        if verbose:
            print('Computing Castelli & Kurucz (2004) passband extinction corrections for %s:%s. This will take a while.' % (self.pbset, self.pbname))

        # Covered wavelengths in the fits tables:
        wavelengths = np.arange(900., 39999.501, 0.5)/1e10 # AA -> m

        for i, model in enumerate(models):
            with fits.open(model) as hdu:
                intensities = hdu[0].data[-1,:]*1e7  # erg/s/cm^2/A -> W/m^3
            spc = np.vstack((wavelengths, intensities))

            model = model[model.rfind('/')+1:] # get relative pathname
            Teff[i] = float(model[1:6])
            logg[i] = float(model[7:9])/10
            abun[i] = float(model[10:12])/10 * (-1 if model[9] == 'M' else 1)

            sel = (spc[0] >= self.ptf_table['wl'][0]) & (spc[0] <= self.ptf_table['wl'][-1])

            wl = spc[0][sel]
            fl = spc[1][sel]

            fl *= self.ptf(wl)
            flP = fl*wl

            # Alambda = np.matmul(libphoebe.CCM89_extinction(wl), M[i])
            Alambda = np.matmul(libphoebe.gordon_extinction(wl), M[i])
            flux_frac = np.exp(-0.9210340371976184*Alambda)             #10**(-0.4*Alambda)

            extinctE[i], extinctP[i] = np.dot([fl/fl.sum(), flP/flP.sum()], flux_frac)

            if verbose:
                sys.stdout.write('\r' + '%0.0f%% done.' % (100*i/(Nmodels-1)))
                sys.stdout.flush()

        if verbose:
            print('')

        # Store axes (Teff, logg, abun) and the full grid of Inorm, with
        # nans where the grid isn't complete.
        self._ck2004_extinct_axes = (np.unique(Teff), np.unique(logg), np.unique(abun), np.unique(Ebv), np.unique(Rv))

        Teff = np.repeat(Teff, Ns)
        logg = np.repeat(logg, Ns)
        abun = np.repeat(abun, Ns)

        self._ck2004_extinct_energy_grid = np.nan*np.ones((len(self._ck2004_extinct_axes[0]), len(self._ck2004_extinct_axes[1]), len(self._ck2004_extinct_axes[2]), len(self._ck2004_extinct_axes[3]), len(self._ck2004_extinct_axes[4]), 1))
        self._ck2004_extinct_photon_grid = np.copy(self._ck2004_extinct_energy_grid)

        flatE = extinctE.flat
        flatP = extinctP.flat

        for i in range(combos):
            t = (Teff[i] == self._ck2004_extinct_axes[0], logg[i] == self._ck2004_extinct_axes[1], abun[i] == self._ck2004_extinct_axes[2], Ebv1[i] == self._ck2004_extinct_axes[3], Rv1[i] == self._ck2004_extinct_axes[4], 0)
            self._ck2004_extinct_energy_grid[t] = flatE[i]
            self._ck2004_extinct_photon_grid[t] = flatP[i]

        if 'ck2004:ext' not in self.content:
            self.content.append('ck2004:ext')

    def compute_phoenix_reddening(self, path, Ebv=None, Rv=None, verbose=False):
        """
        Computes mean effect of reddening (a weighted average) on passband using
        phoenix atmospheres and CCM89 prescription of extinction.

        See also:
        * <phoebe.atmospheres.passbands.Passband.compute_bb_reddening>
        * <phoebe.atmospheres.passbands.Passband.compute_ck2004_reddening>

        Arguments
        ------------
        * `path` (string): path to the directory containing ck2004 SEDs
        * `Ebv` (float or None, optional, default=None): colour discrepancies E(B-V)
        * `Rv` (float or None, optional, default=None): Extinction factor
            (defined at Av / E(B-V) where Av is the visual extinction in magnitudes)
        * `verbose` (bool, optional, default=False): switch to determine whether
            computing progress should be printed on screen
        """

        if Ebv is None:
            Ebv = np.linspace(0., 3., 30)

        if Rv is None:
            Rv = np.linspace(2., 6., 16)

        models = glob.glob(path+'/*fits')
        Nmodels = len(models)

        NEbv = len(Ebv)
        NRv = len(Rv)

        Ns = NEbv*NRv
        combos = Nmodels*Ns

        Ebv1 = np.tile(np.repeat(Ebv, NRv), Nmodels)
        Rv1 = np.tile(Rv, int(combos/NRv))

        # auxilary matrix for storing Ebv and Rv per model
        M = np.rollaxis(np.array([np.split(Ebv1*Rv1, Nmodels), np.split(Ebv1, Nmodels)]), 1)
        M = np.ascontiguousarray(M)

        # Store the length of the filename extensions for parsing:
        offset = len(models[0])-models[0].rfind('.')

        Teff, logg, abun = np.empty(Nmodels), np.empty(Nmodels), np.empty(Nmodels)

        # extinctE , extinctP per model
        extinctE, extinctP = np.empty((Nmodels, Ns)), np.empty((Nmodels, Ns))

        if verbose:
            print('Computing PHOENIX (Husser et al. 2013) passband extinction corrections for %s:%s. This will take a while.' % (self.pbset, self.pbname))

        wavelengths = np.arange(500., 26000.)/1e10 # AA -> m

        for i, model in enumerate(models):
            with fits.open(model) as hdu:
                intensities = hdu[0].data[-1,:]*1e-1
            spc = np.vstack((wavelengths, intensities))

            model = model[model.rfind('/')+1:] # get relative pathname
            Teff[i] = float(model[3:8])
            logg[i] = float(model[9:13])
            abun[i] = float(model[13:17])

            wl = spc[0][(spc[0] >= self.ptf_table['wl'][0]) & (spc[0] <= self.ptf_table['wl'][-1])]
            fl = spc[1][(spc[0] >= self.ptf_table['wl'][0]) & (spc[0] <= self.ptf_table['wl'][-1])]
            fl *= self.ptf(wl)
            flP = fl*wl

            # Alambda = np.matmul(libphoebe.CCM89_extinction(wl), M[i])
            Alambda = np.matmul(libphoebe.gordon_extinction(wl), M[i])
            flux_frac = np.exp(-0.9210340371976184*Alambda)             #10**(-0.4*Alambda)

            extinctE[i], extinctP[i]= np.dot([fl/fl.sum(), flP/flP.sum()], flux_frac)

            if verbose:
                sys.stdout.write('\r' + '%0.0f%% done.' % (100*i/(Nmodels-1)))
                sys.stdout.flush()

        if verbose:
            print('')

        # Store axes (Teff, logg, abun) and the full grid of Inorm, with
        # nans where the grid isn't complete.
        self._phoenix_extinct_axes = (np.unique(Teff), np.unique(logg), np.unique(abun), np.unique(Ebv), np.unique(Rv))

        Teff=np.repeat(Teff, Ns)
        logg=np.repeat(logg, Ns)
        abun=np.repeat(abun, Ns)

        self._phoenix_extinct_energy_grid = np.nan*np.ones((len(self._phoenix_extinct_axes[0]), len(self._phoenix_extinct_axes[1]), len(self._phoenix_extinct_axes[2]), len(self._phoenix_extinct_axes[3]), len(self._phoenix_extinct_axes[4]), 1))
        self._phoenix_extinct_photon_grid = np.copy(self._phoenix_extinct_energy_grid)

        flatE = extinctE.flat
        flatP = extinctP.flat

        for i in range(combos):
            t = (Teff[i] == self._phoenix_extinct_axes[0], logg[i] == self._phoenix_extinct_axes[1], abun[i] == self._phoenix_extinct_axes[2], Ebv1[i] == self._phoenix_extinct_axes[3], Rv1[i] == self._phoenix_extinct_axes[4], 0)
            self._phoenix_extinct_energy_grid[t] = flatE[i]
            self._phoenix_extinct_photon_grid[t] = flatP[i]

        if 'phoenix:ext' not in self.content:
            self.content.append('phoenix:ext')

    def compute_ck2004_response(self, path, verbose=False):
        """
        Computes Castelli & Kurucz (2004) intensities across the entire
        range of model atmospheres.

        Arguments
        -----------
        * `path` (string): path to the directory containing ck2004 SEDs.
        * `verbose` (bool, optional, default=False): switch to determine whether
            computing progress should be printed on screen.
        """

        models = glob.glob(path+'/*fits')
        Nmodels = len(models)

        Teff, logg, abun = np.empty(Nmodels), np.empty(Nmodels), np.empty(Nmodels)
        InormE, InormP = np.empty(Nmodels), np.empty(Nmodels)

        if verbose:
            print('Computing Castelli & Kurucz (2004) normal passband intensities for %s:%s.' % (self.pbset, self.pbname))

        # Covered wavelengths in the fits tables:
        wavelengths = np.arange(900., 39999.501, 0.5)/1e10 # AA -> m

        for i, model in enumerate(models):
            with fits.open(model) as hdu:
                intensities = hdu[0].data[-1,:]*1e7  # erg/s/cm^2/A -> W/m^3
            spc = np.vstack((wavelengths, intensities))

            model = model[model.rfind('/')+1:] # get relative pathname
            Teff[i] = float(model[1:6])
            logg[i] = float(model[7:9])/10
            abun[i] = float(model[10:12])/10 * (-1 if model[9] == 'M' else 1)

            wl = spc[0][(spc[0] >= self.ptf_table['wl'][0]) & (spc[0] <= self.ptf_table['wl'][-1])]
            fl = spc[1][(spc[0] >= self.ptf_table['wl'][0]) & (spc[0] <= self.ptf_table['wl'][-1])]
            fl *= self.ptf(wl)
            flP = fl*wl
            InormE[i] = np.log10(fl.sum()/self.ptf_area*(wl[1]-wl[0]))             # energy-weighted intensity
            InormP[i] = np.log10(flP.sum()/self.ptf_photon_area*(wl[1]-wl[0]))     # photon-weighted intensity
            if verbose:
                sys.stdout.write('\r' + '%0.0f%% done.' % (100*float(i+1)/len(models)))
                sys.stdout.flush()

        if verbose:
            print('')

        # Store axes (Teff, logg, abun) and the full grid of Inorm, with
        # nans where the grid isn't complete.
        self._ck2004_axes = (np.unique(Teff), np.unique(logg), np.unique(abun))

        self._ck2004_photon_grid = np.nan*np.ones((len(self._ck2004_axes[0]), len(self._ck2004_axes[1]), len(self._ck2004_axes[2]), 1))
        self._ck2004_energy_grid = np.nan*np.ones((len(self._ck2004_axes[0]), len(self._ck2004_axes[1]), len(self._ck2004_axes[2]), 1))
        for i, I0 in enumerate(InormE):
            self._ck2004_energy_grid[Teff[i] == self._ck2004_axes[0], logg[i] == self._ck2004_axes[1], abun[i] == self._ck2004_axes[2], 0] = I0
        for i, I0 in enumerate(InormP):
            self._ck2004_photon_grid[Teff[i] == self._ck2004_axes[0], logg[i] == self._ck2004_axes[1], abun[i] == self._ck2004_axes[2], 0] = I0

        # Tried radial basis functions but they were just terrible.
        #~ self._log10_Inorm_ck2004 = interpolate.Rbf(self._ck2004_Teff, self._ck2004_logg, self._ck2004_met, self._ck2004_Inorm, function='linear')

        if 'ck2004:Inorm' not in self.content:
            self.content.append('ck2004:Inorm')

    def compute_phoenix_response(self, path, verbose=False):
        """
        Computes PHOENIX (Husser et al. 2013, A&A 553, 6) intensities across the entire
        range of model atmospheres.

        Arguments
        -----------
        * `path` (string): path to the directory containing ck2004 SEDs.
        * `verbose` (bool, optional, default=False): switch to determine whether
            computing progress should be printed on screen.
        """

        models = glob.glob(path+'/*fits')
        Nmodels = len(models)

        Teff, logg, abun = np.empty(Nmodels), np.empty(Nmodels), np.empty(Nmodels)
        InormE, InormP = np.empty(Nmodels), np.empty(Nmodels)

        if verbose:
            print('Computing PHOENIX (Husser et al. 2013) normal passband intensities for %s:%s.' % (self.pbset, self.pbname))

        wavelengths = np.arange(500., 26000.)/1e10 # AA -> m

        for i, model in enumerate(models):
            with fits.open(model) as hdu:
                intensities = hdu[0].data[-1,:]*1e-1
            spc = np.vstack((wavelengths, intensities))

            model = model[model.rfind('/')+1:] # get relative pathname
            Teff[i] = float(model[3:8])
            logg[i] = float(model[9:13])
            abun[i] = float(model[13:17])

            wl = spc[0][(spc[0] >= self.ptf_table['wl'][0]) & (spc[0] <= self.ptf_table['wl'][-1])]
            fl = spc[1][(spc[0] >= self.ptf_table['wl'][0]) & (spc[0] <= self.ptf_table['wl'][-1])]
            fl *= self.ptf(wl)
            flP = fl*wl
            InormE[i] = np.log10(fl.sum()/self.ptf_area*(wl[1]-wl[0]))             # energy-weighted intensity
            InormP[i] = np.log10(flP.sum()/self.ptf_photon_area*(wl[1]-wl[0]))     # photon-weighted intensity
            if verbose:
                sys.stdout.write('\r' + '%0.0f%% done.' % (100*float(i+1)/len(models)))
                sys.stdout.flush()

        if verbose:
            print('')

        # Store axes (Teff, logg, abun) and the full grid of Inorm, with
        # nans where the grid isn't complete.
        self._phoenix_axes = (np.unique(Teff), np.unique(logg), np.unique(abun))

        self._phoenix_energy_grid = np.nan*np.ones((len(self._phoenix_axes[0]), len(self._phoenix_axes[1]), len(self._phoenix_axes[2]), 1))
        self._phoenix_photon_grid = np.nan*np.ones((len(self._phoenix_axes[0]), len(self._phoenix_axes[1]), len(self._phoenix_axes[2]), 1))
        for i, I0 in enumerate(InormE):
            self._phoenix_energy_grid[Teff[i] == self._phoenix_axes[0], logg[i] == self._phoenix_axes[1], abun[i] == self._phoenix_axes[2], 0] = I0
        for i, I0 in enumerate(InormP):
            self._phoenix_photon_grid[Teff[i] == self._phoenix_axes[0], logg[i] == self._phoenix_axes[1], abun[i] == self._phoenix_axes[2], 0] = I0

        # Tried radial basis functions but they were just terrible.
        #~ self._log10_Inorm_ck2004 = interpolate.Rbf(self._ck2004_Teff, self._ck2004_logg, self._ck2004_met, self._ck2004_Inorm, function='linear')

        if 'phoenix:Inorm' not in self.content:
            self.content.append('phoenix:Inorm')

    def _rescale_phoenix_intensities(self, mu_interp, mu_phoenix, intensity_phoenix):
        '''
        Rescales spherical PHOENIX intensities so that I(mu=0) = 0.

        The PHOENIX intensities are recomputed past the inflection point using the
        tangent in the inflection point. Mus are rescaled such that mu=0 where the
        tangent in the inflection point intersects the x-axis.
        '''

        def mu_inflection(mu, g2):

            argmax = np.argmin(g2)
            argmin = np.argmax(g2[0:argmax])
            g2_interp = interpolate.interp1d(g2[argmin:argmax+1], mu[argmin:argmax+1])

            return g2_interp([0.])

        def tangent(mu, s, g, mu_infl):
            g1_interp = interpolate.interp1d(mu, g)
            s_interp = interpolate.interp1d(mu, s)

            g_infl = g1_interp(mu_infl)
            s_infl = s_interp(mu_infl)

            n_tan = s_infl - g_infl*mu_infl

            return [g_infl, n_tan]

        # compute the first and second gradient

        g1 = np.gradient(intensity_phoenix, mu_phoenix)
        g2 = np.gradient(g1, mu_phoenix)

        # compute the inflection point and tangent
        mu_infl = mu_inflection(mu_phoenix, g2)
        k, n = tangent(mu_phoenix, intensity_phoenix, g1, mu_infl)

        # compute mu where y-tangent = 0
        mu0 = -n/k
        # for passbands that don't trail off, mu0 is always < 0
        # using that to catch them and cut off at the true end-point
        if mu0 < mu_phoenix.min():
            mu_infl = mu_phoenix.min()
            mu0 = mu_phoenix.min()

        # recompute intensities
        intensity_phoenix[mu_phoenix<mu_infl] = k*mu_phoenix[mu_phoenix<mu_infl] + n
        intensity_phoenix[mu_phoenix<mu0] = 0.

        # renormalize mus on 0 to 1
        mus_norm = np.cos(np.pi/2*np.arccos(mu_phoenix)/np.arccos(mu0))

        # if there's no real cut-off, above computation returns lowest value approx. 1e-17
        # additional rescaling to fix mus_norm on range [0.,1.]
        if mus_norm.min() > 0:
            mus_norm = (mus_norm - mus_norm.min())/(mus_norm-mus_norm.min()).max()

        # interpolate intensities in user-provided mus
        intensity_interp = interpolate.interp1d(mus_norm, intensity_phoenix)
        return intensity_interp(mu_interp)

    def compute_ck2004_intensities(self, path, particular=None, verbose=False):
        """
        Computes direction-dependent passband intensities using Castelli & Kurucz (2004)
        model atmospheres.

        Arguments
        -----------
        * `path` (string): path to the directory with SEDs in FITS format.
        * `particular` (string, optional, default=None): particular file in
            `path` to be processed; if None, all files in the directory are
            processed.
        * `verbose` (bool, optional, default=False): set to True to display
            progress in the terminal.
        """

        if verbose:
            print('Computing Castelli & Kurucz (2004) specific passband intensities for %s:%s.' % (self.pbset, self.pbname))

        models = glob.glob(path+'/*fits')
        Nmodels = len(models)

        mus = np.array([0., 0.001, 0.002, 0.003, 0.005, 0.01 , 0.015, 0.02 , 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])

        Teff, logg, abun = np.empty(Nmodels), np.empty(Nmodels), np.empty(Nmodels)

        ImuE, ImuP = np.empty(Nmodels*len(mus)), np.empty(Nmodels*len(mus))
        # boostingE, boostingP = np.empty(Nmodels), np.empty(Nmodels)

        wavelengths = np.arange(900., 39999.501, 0.5)/1e10 # AA -> m

        keep = (wavelengths >= self.ptf_table['wl'][0]) & (wavelengths <= self.ptf_table['wl'][-1])
        wl = wavelengths[keep]
        dwl = wl[1]-wl[0]

        for i, model in enumerate(models):
            with fits.open(model) as hdu:
                intensities = hdu[0].data*1e7 # erg/s/cm^2/A -> W/m^3
                # FIXME: the line below is not needed when atm gets updated:
                intensities[0,:] = np.ones(len(wavelengths))

                # trim the spectrum at passband limits:
                intensities = intensities[:,keep]

            model = model[model.rfind('/')+1:] # get relative pathname
            Teff[i] = float(model[1:6])
            logg[i] = float(model[7:9])/10
            abun[i] = float(model[10:12])/10 * (-1 if model[9] == 'M' else 1)

            flE = self.ptf(wl)*intensities
            flEint = flE.sum(axis=1)

            flP = wl*flE
            flPint = flP.sum(axis=1)

            ImuE[i*len(mus):(i+1)*len(mus)] = np.log10(flEint/self.ptf_area*dwl)        # energy-weighted intensity
            ImuP[i*len(mus):(i+1)*len(mus)] = np.log10(flPint/self.ptf_photon_area*dwl) # photon-weighted intensity

            if verbose:
                sys.stdout.write('\r' + '%0.0f%% done.' % (100*float(i+1)/len(models)))
                sys.stdout.flush()

        if verbose:
            print('')

            # for cmi, cmu in enumerate(mus):
            #     fl = intensities[cmi,:]

                # make a log-scale copy for boosting and fit a Legendre
                # polynomial to the Imu envelope by way of sigma clipping;
                # then compute a Legendre series derivative to get the
                # boosting index; we only take positive fluxes to keep the
                # log well defined.

                # lnwl = np.log(wl[fl > 0])
                # lnfl = np.log(fl[fl > 0]) + 5*lnwl

                # First Legendre fit to the data:
                # envelope = np.polynomial.legendre.legfit(lnwl, lnfl, 5)
                # continuum = np.polynomial.legendre.legval(lnwl, envelope)
                # diff = lnfl-continuum
                # sigma = np.std(diff)
                # clipped = (diff > -sigma)

                # Sigma clip to get the continuum:
                # while True:
                #     Npts = clipped.sum()
                #     envelope = np.polynomial.legendre.legfit(lnwl[clipped], lnfl[clipped], 5)
                #     continuum = np.polynomial.legendre.legval(lnwl, envelope)
                #     diff = lnfl-continuum

                    # clipping will sometimes unclip already clipped points
                    # because the fit is slightly different, which can lead
                    # to infinite loops. To prevent that, we never allow
                    # clipped points to be resurrected, which is achieved
                    # by the following bitwise condition (array comparison):
                #     clipped = clipped & (diff > -sigma)

                #     if clipped.sum() == Npts:
                #         break

                # derivative = np.polynomial.legendre.legder(envelope, 1)
                # boosting_index = np.polynomial.legendre.legval(lnwl, derivative)

                # calculate energy (E) and photon (P) weighted fluxes and
                # their integrals.

                # calculate mean boosting coefficient and use it to get
                # boosting factors for energy (E) and photon (P) weighted
                # fluxes.

                # boostE = (flE[fl > 0]*boosting_index).sum()/flEint
                # boostP = (flP[fl > 0]*boosting_index).sum()/flPint
                # boostingE[i] = boostE
                # boostingP[i] = boostP

        self._ck2004_intensity_axes = (np.unique(Teff), np.unique(logg), np.unique(abun), np.unique(mus))
        self._ck2004_Imu_energy_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1))
        self._ck2004_Imu_photon_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1))
        # self._ck2004_boosting_energy_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1))
        # self._ck2004_boosting_photon_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1))

        # Set the limb (mu=0) to 0; in log this actually means
        # flux=1W/m2, but for all practical purposes that is still 0.
        self._ck2004_Imu_energy_grid[:,:,:,0,:] = 0.0
        self._ck2004_Imu_photon_grid[:,:,:,0,:] = 0.0
        # self._ck2004_boosting_energy_grid[:,:,:,0,:] = 0.0
        # self._ck2004_boosting_photon_grid[:,:,:,0,:] = 0.0

        for i, Imu in enumerate(ImuE):
            self._ck2004_Imu_energy_grid[Teff[int(i/len(mus))] == self._ck2004_intensity_axes[0], logg[int(i/len(mus))] == self._ck2004_intensity_axes[1], abun[int(i/len(mus))] == self._ck2004_intensity_axes[2], mus[i%len(mus)] == self._ck2004_intensity_axes[3], 0] = Imu
        for i, Imu in enumerate(ImuP):
            self._ck2004_Imu_photon_grid[Teff[int(i/len(mus))] == self._ck2004_intensity_axes[0], logg[int(i/len(mus))] == self._ck2004_intensity_axes[1], abun[int(i/len(mus))] == self._ck2004_intensity_axes[2], mus[i%len(mus)] == self._ck2004_intensity_axes[3], 0] = Imu
        # for i, Bavg in enumerate(boostingE):
        #     self._ck2004_boosting_energy_grid[Teff[i] == self._ck2004_intensity_axes[0], logg[i] == self._ck2004_intensity_axes[1], abun[i] == self._ck2004_intensity_axes[2], mu[i] == self._ck2004_intensity_axes[3], 0] = Bavg
        # for i, Bavg in enumerate(boostingP):
        #     self._ck2004_boosting_photon_grid[Teff[i] == self._ck2004_intensity_axes[0], logg[i] == self._ck2004_intensity_axes[1], abun[i] == self._ck2004_intensity_axes[2], mu[i] == self._ck2004_intensity_axes[3], 0] = Bavg

        if 'ck2004:Imu' not in self.content:
            self.content.append('ck2004:Imu')

    def compute_phoenix_intensities(self, path, particular=None, verbose=False):
        """
        Computes direction-dependent passband intensities using spherical
        PHOENIX (Husser et al. 2013) model atmospheres.

        Arguments
        -----------
        * `path` (string): path to the directory with SEDs in FITS format.
        * `particular` (string, optional, default=None): particular file in
            `path` to be processed; if None, all files in the directory are
            processed.
        * `verbose` (bool, optional, default=False): set to True to display
            progress in the terminal.
        """

        if verbose:
            print('Computing PHOENIX (Husser et al. 2013) specific passband intensities for %s:%s.' % (self.pbset, self.pbname))

        models = glob.glob(path+'/*fits')
        Nmodels = len(models)

        # the values of mu are hard-coded to the ck2004 values for 1-to-1 comparison:
        mu = np.array([0., 0.001, 0.002, 0.003, 0.005, 0.01 , 0.015, 0.02 , 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])

        Teff, logg, abun = np.empty(Nmodels), np.empty(Nmodels), np.empty(Nmodels)

        ImuE, ImuP = np.empty(Nmodels*len(mu)), np.empty(Nmodels*len(mu))
        # boostingE, boostingP = np.empty(Nmodels), np.empty(Nmodels)

        wavelengths = np.arange(500., 26000.)/1e10 # AA -> m
        keep = (wavelengths >= self.ptf_table['wl'][0]) & (wavelengths <= self.ptf_table['wl'][-1])
        wl = wavelengths[keep]
        dwl = wl[1]-wl[0]

        for i, model in enumerate(models):
            with fits.open(model) as hdu:
                mus = hdu[1].data
                intensities = hdu[0].data*1e-1

                # trim the spectrum at passband limits:
                intensities = intensities[:,keep]

            model = model[model.rfind('/')+1:] # get relative pathname
            Teff[i] = float(model[3:8])
            logg[i] = float(model[9:13])
            abun[i] = float(model[13:17])

            flE = self.ptf(wl)*intensities
            flEint = flE.sum(axis=1)
            flEint = self._rescale_phoenix_intensities(mu, mus, flEint)

            flP = wl*flE
            flPint = flP.sum(axis=1)
            flPint = self._rescale_phoenix_intensities(mu, mus, flPint)

            ImuE[i*len(mu):(i+1)*len(mu)] = np.log10(flEint/self.ptf_area*dwl)        # energy-weighted intensity
            ImuP[i*len(mu):(i+1)*len(mu)] = np.log10(flPint/self.ptf_photon_area*dwl) # photon-weighted intensity

            if verbose:
                sys.stdout.write('\r' + '%0.0f%% done.' % (100*float(i+1)/len(models)))
                sys.stdout.flush()

        if verbose:
            print('')

            # for cmi, cmu in enumerate(mus):
            #     fl = intensities[cmi,:]

                # make a log-scale copy for boosting and fit a Legendre
                # polynomial to the Imu envelope by way of sigma clipping;
                # then compute a Legendre series derivative to get the
                # boosting index; we only take positive fluxes to keep the
                # log well defined.

                # lnwl = np.log(wl[fl > 0])
                # lnfl = np.log(fl[fl > 0]) + 5*lnwl

                # First Legendre fit to the data:
                # envelope = np.polynomial.legendre.legfit(lnwl, lnfl, 5)
                # continuum = np.polynomial.legendre.legval(lnwl, envelope)
                # diff = lnfl-continuum
                # sigma = np.std(diff)
                # clipped = (diff > -sigma)

                # Sigma clip to get the continuum:
                # while True:
                #     Npts = clipped.sum()
                #     envelope = np.polynomial.legendre.legfit(lnwl[clipped], lnfl[clipped], 5)
                #     continuum = np.polynomial.legendre.legval(lnwl, envelope)
                #     diff = lnfl-continuum

                    # clipping will sometimes unclip already clipped points
                    # because the fit is slightly different, which can lead
                    # to infinite loops. To prevent that, we never allow
                    # clipped points to be resurrected, which is achieved
                    # by the following bitwise condition (array comparison):
                #     clipped = clipped & (diff > -sigma)

                #     if clipped.sum() == Npts:
                #         break

                # derivative = np.polynomial.legendre.legder(envelope, 1)
                # boosting_index = np.polynomial.legendre.legval(lnwl, derivative)

                # calculate energy (E) and photon (P) weighted fluxes and
                # their integrals.

                # calculate mean boosting coefficient and use it to get
                # boosting factors for energy (E) and photon (P) weighted
                # fluxes.

                # boostE = (flE[fl > 0]*boosting_index).sum()/flEint
                # boostP = (flP[fl > 0]*boosting_index).sum()/flPint
                # boostingE[i] = boostE
                # boostingP[i] = boostP



        # Store axes (Teff, logg, abun, mu) and the full grid of Imu,
        # with nans where the grid isn't complete. Imu-s come in two
        # flavors: energy-weighted intensities and photon-weighted
        # intensities, based on the detector used.

        self._phoenix_intensity_axes = (np.unique(Teff), np.unique(logg), np.unique(abun), np.unique(mu))
        self._phoenix_Imu_energy_grid = np.nan*np.ones((len(self._phoenix_intensity_axes[0]), len(self._phoenix_intensity_axes[1]), len(self._phoenix_intensity_axes[2]), len(self._phoenix_intensity_axes[3]), 1))
        self._phoenix_Imu_photon_grid = np.nan*np.ones((len(self._phoenix_intensity_axes[0]), len(self._phoenix_intensity_axes[1]), len(self._phoenix_intensity_axes[2]), len(self._phoenix_intensity_axes[3]), 1))
        # self._ck2004_boosting_energy_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1))
        # self._ck2004_boosting_photon_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1))

        # Set the limb (mu=0) to 0; in log this actually means
        # flux=1W/m2, but for all practical purposes that is still 0.
        # self._ck2004_Imu_energy_grid[:,:,:,0,:] = 0.0
        # self._ck2004_Imu_photon_grid[:,:,:,0,:] = 0.0
        # self._ck2004_boosting_energy_grid[:,:,:,0,:] = 0.0
        # self._ck2004_boosting_photon_grid[:,:,:,0,:] = 0.0

        for i, Imu in enumerate(ImuE):
            self._phoenix_Imu_energy_grid[Teff[int(i/len(mu))] == self._phoenix_intensity_axes[0], logg[int(i/len(mu))] == self._phoenix_intensity_axes[1], abun[int(i/len(mu))] == self._phoenix_intensity_axes[2], mu[i%len(mu)] == self._phoenix_intensity_axes[3], 0] = Imu
        for i, Imu in enumerate(ImuP):
            self._phoenix_Imu_photon_grid[Teff[int(i/len(mu))] == self._phoenix_intensity_axes[0], logg[int(i/len(mu))] == self._phoenix_intensity_axes[1], abun[int(i/len(mu))] == self._phoenix_intensity_axes[2], mu[i%len(mu)] == self._phoenix_intensity_axes[3], 0] = Imu
        # for i, Bavg in enumerate(boostingE):
        #     self._ck2004_boosting_energy_grid[Teff[i] == self._ck2004_intensity_axes[0], logg[i] == self._ck2004_intensity_axes[1], abun[i] == self._ck2004_intensity_axes[2], mu[i] == self._ck2004_intensity_axes[3], 0] = Bavg
        # for i, Bavg in enumerate(boostingP):
        #     self._ck2004_boosting_photon_grid[Teff[i] == self._ck2004_intensity_axes[0], logg[i] == self._ck2004_intensity_axes[1], abun[i] == self._ck2004_intensity_axes[2], mu[i] == self._ck2004_intensity_axes[3], 0] = Bavg

        if 'phoenix:Imu' not in self.content:
            self.content.append('phoenix:Imu')

    def _ldlaw_lin(self, mu, xl):
        return 1.0-xl*(1-mu)

    def _ldlaw_log(self, mu, xl, yl):
        return 1.0-xl*(1-mu)-yl*mu*np.log(mu+1e-6)

    def _ldlaw_sqrt(self, mu, xl, yl):
        return 1.0-xl*(1-mu)-yl*(1.0-np.sqrt(mu))

    def _ldlaw_quad(self, mu, xl, yl):
        return 1.0-xl*(1.0-mu)-yl*(1.0-mu)*(1.0-mu)

    def _ldlaw_nonlin(self, mu, c1, c2, c3, c4):
        return 1.0-c1*(1.0-np.sqrt(mu))-c2*(1.0-mu)-c3*(1.0-mu*np.sqrt(mu))-c4*(1.0-mu*mu)

    def compute_ck2004_ldcoeffs(self, weighting='uniform', plot_diagnostics=False):
        """
        Computes limb darkening coefficients for linear, log, square root,
        quadratic and power laws.

        Arguments
        ----------
        * `weighting` (string, optional, default='uniform'): determines how data
            points should be weighted.
            * 'uniform':  do not apply any per-point weighting
            * 'interval': apply weighting based on the interval widths
        """
        if 'ck2004:Imu' not in self.content:
            print('Castelli & Kurucz (2004) intensities are not computed yet. Please compute those first.')
            return None

        self._ck2004_ld_energy_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), 11))
        self._ck2004_ld_photon_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), 11))
        mus = self._ck2004_intensity_axes[3] # starts with 0
        if weighting == 'uniform':
            sigma = np.ones(len(mus))
        elif weighting == 'interval':
            delta = np.concatenate( (np.array((mus[1]-mus[0],)), mus[1:]-mus[:-1]) )
            sigma = 1./np.sqrt(delta)
        else:
            print('Weighting scheme \'%s\' is unsupported. Please choose among [\'uniform\', \'interval\']')
            return None

        for Tindex in range(len(self._ck2004_intensity_axes[0])):
            for lindex in range(len(self._ck2004_intensity_axes[1])):
                for mindex in range(len(self._ck2004_intensity_axes[2])):
                    IsE = 10**self._ck2004_Imu_energy_grid[Tindex,lindex,mindex,:].flatten()
                    fEmask = np.isfinite(IsE)
                    if len(IsE[fEmask]) <= 1:
                        continue
                    IsE /= IsE[fEmask][-1]

                    cElin,  pcov = cfit(f=self._ldlaw_lin,    xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5])
                    cElog,  pcov = cfit(f=self._ldlaw_log,    xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cEsqrt, pcov = cfit(f=self._ldlaw_sqrt,   xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cEquad, pcov = cfit(f=self._ldlaw_quad,   xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cEnlin, pcov = cfit(f=self._ldlaw_nonlin, xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5, 0.5, 0.5, 0.5])
                    self._ck2004_ld_energy_grid[Tindex, lindex, mindex] = np.hstack((cElin, cElog, cEsqrt, cEquad, cEnlin))

                    IsP = 10**self._ck2004_Imu_photon_grid[Tindex,lindex,mindex,:].flatten()
                    fPmask = np.isfinite(IsP)
                    IsP /= IsP[fPmask][-1]

                    cPlin,  pcov = cfit(f=self._ldlaw_lin,    xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5])
                    cPlog,  pcov = cfit(f=self._ldlaw_log,    xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cPsqrt, pcov = cfit(f=self._ldlaw_sqrt,   xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cPquad, pcov = cfit(f=self._ldlaw_quad,   xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cPnlin, pcov = cfit(f=self._ldlaw_nonlin, xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5, 0.5, 0.5, 0.5])
                    self._ck2004_ld_photon_grid[Tindex, lindex, mindex] = np.hstack((cPlin, cPlog, cPsqrt, cPquad, cPnlin))

                    if plot_diagnostics:
                        if Tindex == 10 and lindex == 9 and mindex == 5:
                            print(self._ck2004_intensity_axes[0][Tindex], self._ck2004_intensity_axes[1][lindex], self._ck2004_intensity_axes[2][mindex])
                            print(mus, IsE)
                            print(cElin, cElog, cEsqrt)
                            import matplotlib.pyplot as plt
                            plt.plot(mus[fEmask], IsE[fEmask], 'bo')
                            plt.plot(mus[fEmask], self._ldlaw_lin(mus[fEmask], *cElin), 'r-')
                            plt.plot(mus[fEmask], self._ldlaw_log(mus[fEmask], *cElog), 'g-')
                            plt.plot(mus[fEmask], self._ldlaw_sqrt(mus[fEmask], *cEsqrt), 'y-')
                            plt.plot(mus[fEmask], self._ldlaw_quad(mus[fEmask], *cEquad), 'm-')
                            plt.plot(mus[fEmask], self._ldlaw_nonlin(mus[fEmask], *cEnlin), 'k-')
                            plt.show()

        if 'ck2004:ld' not in self.content:
            self.content.append('ck2004:ld')

    def compute_phoenix_ldcoeffs(self, weighting='uniform', plot_diagnostics=False):
        """
        Computes limb darkening coefficients from PHOENIX atmospheres for the linear,
        log, square root, quadratic and power laws.

        Arguments
        ----------
        * `weighting` (string, optional, default='uniform'): determines how data
            points should be weighted.
            * 'uniform':  do not apply any per-point weighting
            * 'interval': apply weighting based on the interval widths
        """
        if 'phoenix:Imu' not in self.content:
            print('PHOENIX (Husser et al. 2013) intensities are not computed yet. Please compute those first.')
            return None

        self._phoenix_ld_energy_grid = np.nan*np.ones((len(self._phoenix_intensity_axes[0]), len(self._phoenix_intensity_axes[1]), len(self._phoenix_intensity_axes[2]), 11))
        self._phoenix_ld_photon_grid = np.nan*np.ones((len(self._phoenix_intensity_axes[0]), len(self._phoenix_intensity_axes[1]), len(self._phoenix_intensity_axes[2]), 11))
        mus = self._phoenix_intensity_axes[3] # starts with 0
        if weighting == 'uniform':
            sigma = np.ones(len(mus))
        elif weighting == 'interval':
            delta = np.concatenate( (np.array((mus[1]-mus[0],)), mus[1:]-mus[:-1]) )
            sigma = 1./np.sqrt(delta)
        else:
            print('Weighting scheme \'%s\' is unsupported. Please choose among [\'uniform\', \'interval\']')
            return None

        for Tindex in range(len(self._phoenix_intensity_axes[0])):
            for lindex in range(len(self._phoenix_intensity_axes[1])):
                for mindex in range(len(self._phoenix_intensity_axes[2])):
                    IsE = 10**self._phoenix_Imu_energy_grid[Tindex,lindex,mindex,:].flatten()
                    fEmask = np.isfinite(IsE)
                    if len(IsE[fEmask]) <= 1:
                        continue
                    IsE /= IsE[fEmask][-1]

                    cElin,  pcov = cfit(f=self._ldlaw_lin,    xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma, p0=[0.5])
                    cElog,  pcov = cfit(f=self._ldlaw_log,    xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma, p0=[0.5, 0.5])
                    cEsqrt, pcov = cfit(f=self._ldlaw_sqrt,   xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma, p0=[0.5, 0.5])
                    cEquad, pcov = cfit(f=self._ldlaw_quad,   xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma, p0=[0.5, 0.5])
                    cEnlin, pcov = cfit(f=self._ldlaw_nonlin, xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma, p0=[0.5, 0.5, 0.5, 0.5])
                    self._phoenix_ld_energy_grid[Tindex, lindex, mindex] = np.hstack((cElin, cElog, cEsqrt, cEquad, cEnlin))

                    IsP = 10**self._phoenix_Imu_photon_grid[Tindex,lindex,mindex,:].flatten()
                    fPmask = np.isfinite(IsP)
                    IsP /= IsP[fPmask][-1]

                    cPlin,  pcov = cfit(f=self._ldlaw_lin,    xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma, p0=[0.5])
                    cPlog,  pcov = cfit(f=self._ldlaw_log,    xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma, p0=[0.5, 0.5])
                    cPsqrt, pcov = cfit(f=self._ldlaw_sqrt,   xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma, p0=[0.5, 0.5])
                    cPquad, pcov = cfit(f=self._ldlaw_quad,   xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma, p0=[0.5, 0.5])
                    cPnlin, pcov = cfit(f=self._ldlaw_nonlin, xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma, p0=[0.5, 0.5, 0.5, 0.5])
                    self._phoenix_ld_photon_grid[Tindex, lindex, mindex] = np.hstack((cPlin, cPlog, cPsqrt, cPquad, cPnlin))

                    if plot_diagnostics:
                        if Tindex == 10 and lindex == 9 and mindex == 5:
                            print(self._phoenix_intensity_axes[0][Tindex], self._phoenix_intensity_axes[1][lindex], self._phoenix_intensity_axes[2][mindex])
                            print(mus, IsE)
                            print(cElin, cElog, cEsqrt)
                            import matplotlib.pyplot as plt
                            plt.plot(mus[fEmask], IsE[fEmask], 'bo')
                            plt.plot(mus[fEmask], self._ldlaw_lin(mus[fEmask], *cElin), 'r-')
                            plt.plot(mus[fEmask], self._ldlaw_log(mus[fEmask], *cElog), 'g-')
                            plt.plot(mus[fEmask], self._ldlaw_sqrt(mus[fEmask], *cEsqrt), 'y-')
                            plt.plot(mus[fEmask], self._ldlaw_quad(mus[fEmask], *cEquad), 'm-')
                            plt.plot(mus[fEmask], self._ldlaw_nonlin(mus[fEmask], *cEnlin), 'k-')
                            plt.show()

        if 'phoenix:ld' not in self.content:
            self.content.append('phoenix:ld')

    def export_phoenix_atmtab(self):
        """
        Exports PHOENIX intensity table to a PHOEBE legacy compatible format.
        """

        teffs = self._phoenix_axes[0]
        tlow, tup = teffs[0], teffs[-1]
        trel = (teffs-tlow)/(tup-tlow)

        for abun in range(len(self._phoenix_axes[2])):
            for logg in range(len(self._phoenix_axes[1])):
                logI = self._phoenix_energy_grid[:,logg,abun,0]+1 # +1 to take care of WD units

                # find the last non-nan value:
                if np.isnan(logI).sum() > 0:
                    imax = len(teffs)-np.where(~np.isnan(logI[::-1]))[0][0]

                    # interpolate any in-between nans:
                    missing, xs = np.isnan(logI[:imax]), lambda z: z.nonzero()[0]
                    logI[:imax][missing] = np.interp(xs(missing), xs(~missing), logI[:imax][~missing])
                else:
                    imax = len(teffs)

                Cl = np.polynomial.legendre.legfit(trel[:imax], logI[:imax], 9)

                print('%8.1f %7.1f % 16.9E % 16.9E % 16.9E % 16.9E % 16.9E % 16.9E % 16.9E % 16.9E % 16.9E % 16.9E' % (teffs[0], teffs[imax-1], Cl[0], Cl[1], Cl[2], Cl[3], Cl[4], Cl[5], Cl[6], Cl[7], Cl[8], Cl[9]))

    def export_legacy_ldcoeffs(self, models, atm='ck2004', filename=None, photon_weighted=True):
        """
        Exports CK2004 limb darkening coefficients to a PHOEBE legacy
        compatible format.

        Arguments
        -----------
        * `models` (string): the path (including the filename) of legacy's
            models.list
        * `atm` (string, default='ck2004'): atmosphere model, 'ck2004' or 'phoenix'
        * `filename` (string, optional, default=None): output filename for
            storing the table
        * `photon_weighted` (bool, optional, default=True): photon/energy switch
        """

        if atm == 'ck2004' and photon_weighted:
            axes = self._ck2004_intensity_axes
            grid = self._ck2004_ld_photon_grid
        elif atm == 'phoenix' and photon_weighted:
            axes = self._phoenix_intensity_axes
            grid = self._phoenix_ld_photon_grid
        elif atm == 'ck2004' and not photon_weighted:
            axes = self._ck2004_intensity_axes
            grid = self._ck2004_ld_energy_grid
        elif atm == 'phoenix' and not photon_weighted:
            axes = self._phoenix_intensity_axes
            grid = self._phoenix_ld_energy_grid
        else:
            print('atmosphere model %s cannot be exported.' % atm)
            return None

        if filename is not None:
            import time
            f = open(filename, 'w')
            f.write('# PASS_SET  %s\n' % self.pbset)
            f.write('# PASSBAND  %s\n' % self.pbname)
            f.write('# VERSION   1.0\n\n')
            f.write('# Exported from PHOEBE-2 passband on %s\n' % (time.ctime()))
            f.write('# The coefficients are computed for the %s-weighted regime from %s atmospheres.\n\n' % ('photon' if photon_weighted else 'energy', atm))

        mods = np.loadtxt(models)
        for mod in mods:
            Tindex = np.argwhere(axes[0] == mod[0])[0][0]
            lindex = np.argwhere(axes[1] == mod[1]/10)[0][0]
            mindex = np.argwhere(axes[2] == mod[2]/10)[0][0]
            if filename is None:
                print('%6.3f '*11 % tuple(grid[Tindex, lindex, mindex].tolist()))
            else:
                f.write(('%6.3f '*11+'\n') % tuple(grid[Tindex, lindex, mindex].tolist()))

        if filename is not None:
            f.close()

    def compute_ck2004_ldints(self):
        """
        Computes integrated limb darkening profiles for ck2004 atmospheres.
        These are used for intensity-to-flux transformations. The evaluated
        integral is:

        ldint = 2 \int_0^1 Imu mu dmu
        """

        if 'ck2004:Imu' not in self.content:
            print('Castelli & Kurucz (2004) intensities are not computed yet. Please compute those first.')
            return None

        ldaxes = self._ck2004_intensity_axes
        ldtable = self._ck2004_Imu_energy_grid
        pldtable = self._ck2004_Imu_photon_grid

        self._ck2004_ldint_energy_grid = np.nan*np.ones((len(ldaxes[0]), len(ldaxes[1]), len(ldaxes[2]), 1))
        self._ck2004_ldint_photon_grid = np.nan*np.ones((len(ldaxes[0]), len(ldaxes[1]), len(ldaxes[2]), 1))

        mu = ldaxes[3]
        Imu = 10**ldtable[:,:,:,:]/10**ldtable[:,:,:,-1:]
        pImu = 10**pldtable[:,:,:,:]/10**pldtable[:,:,:,-1:]

        # To compute the fluxes, we need to evaluate \int_0^1 2pi Imu mu dmu.

        for a in range(len(ldaxes[0])):
            for b in range(len(ldaxes[1])):
                for c in range(len(ldaxes[2])):

                    ldint = 0.0
                    pldint = 0.0
                    for i in range(len(mu)-1):
                        ki = (Imu[a,b,c,i+1]-Imu[a,b,c,i])/(mu[i+1]-mu[i])
                        ni = Imu[a,b,c,i]-ki*mu[i]
                        ldint += ki/3*(mu[i+1]**3-mu[i]**3) + ni/2*(mu[i+1]**2-mu[i]**2)

                        pki = (pImu[a,b,c,i+1]-pImu[a,b,c,i])/(mu[i+1]-mu[i])
                        pni = pImu[a,b,c,i]-pki*mu[i]
                        pldint += pki/3*(mu[i+1]**3-mu[i]**3) + pni/2*(mu[i+1]**2-mu[i]**2)

                    self._ck2004_ldint_energy_grid[a,b,c] = 2*ldint
                    self._ck2004_ldint_photon_grid[a,b,c] = 2*pldint

        if 'ck2004:ldint' not in self.content:
            self.content.append('ck2004:ldint')

    def compute_phoenix_ldints(self):
        """
        Computes integrated limb darkening profiles for PHOENIX atmospheres.
        These are used for intensity-to-flux transformations. The evaluated
        integral is:

        ldint = 2 \pi \int_0^1 Imu mu dmu
        """

        if 'phoenix:Imu' not in self.content:
            print('PHOENIX (Husser et al. 2013) intensities are not computed yet. Please compute those first.')
            return None

        ldaxes = self._phoenix_intensity_axes
        ldtable = self._phoenix_Imu_energy_grid
        pldtable = self._phoenix_Imu_photon_grid

        self._phoenix_ldint_energy_grid = np.nan*np.ones((len(ldaxes[0]), len(ldaxes[1]), len(ldaxes[2]), 1))
        self._phoenix_ldint_photon_grid = np.nan*np.ones((len(ldaxes[0]), len(ldaxes[1]), len(ldaxes[2]), 1))

        mu = ldaxes[3]
        Imu = 10**ldtable[:,:,:,:]/10**ldtable[:,:,:,-1:]
        pImu = 10**pldtable[:,:,:,:]/10**pldtable[:,:,:,-1:]

        # To compute the fluxes, we need to evaluate \int_0^1 2pi Imu mu dmu.

        for a in range(len(ldaxes[0])):
            for b in range(len(ldaxes[1])):
                for c in range(len(ldaxes[2])):

                    ldint = 0.0
                    pldint = 0.0
                    for i in range(len(mu)-1):
                        ki = (Imu[a,b,c,i+1]-Imu[a,b,c,i])/(mu[i+1]-mu[i])
                        ni = Imu[a,b,c,i]-ki*mu[i]
                        ldint += ki/3*(mu[i+1]**3-mu[i]**3) + ni/2*(mu[i+1]**2-mu[i]**2)

                        pki = (pImu[a,b,c,i+1]-pImu[a,b,c,i])/(mu[i+1]-mu[i])
                        pni = pImu[a,b,c,i]-pki*mu[i]
                        pldint += pki/3*(mu[i+1]**3-mu[i]**3) + pni/2*(mu[i+1]**2-mu[i]**2)

                    self._phoenix_ldint_energy_grid[a,b,c] = 2*ldint
                    self._phoenix_ldint_photon_grid[a,b,c] = 2*pldint

        if 'phoenix:ldint' not in self.content:
            self.content.append('phoenix:ldint')

    def interpolate_ldcoeffs(self, Teff=5772., logg=4.43, abun=0.0,
                                    ldatm='ck2004', ld_func='power',
                                    photon_weighted=False):
        """
        Interpolate the passband-stored table of LD model coefficients.

        Arguments
        ------------
        * `Teff` (float or array, default=5772): effective temperature
        * `logg` (float or array, default=4.43): surface gravity in cgs
        * `abun` (float or array, default=0.0): log-abundance in solar log-abundances
        * `ldatm` (string, default='ck2004'): limb darkening table: 'ck2004' or 'phoenix'
        * `ld_func` (string, default='power'): limb darkening fitting function: 'linear',
          'logarithmic', 'square_root', 'quadratic', 'power' or 'all'
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        --------
        * (list or None) list of limb-darkening coefficients or None if 'ck2004:ld'
            is not available in <phoebe.atmospheres.passbands.Passband.content>
            (see also <phoebe.atmospheres.passbands.Passband.compute_ck2004_ldcoeffs>)
            or if `ld_func` is not recognized.
        """

        if ldatm == 'ck2004' and 'ck2004:ld' not in self.content:
            print('Castelli & Kurucz (2004) limb darkening coefficients are not computed yet. Please compute those first.')
            return None

        if ldatm == 'phoenix' and 'phoenix:ld' not in self.content:
            print('PHOENIX (Husser et al. 2013) limb darkening coefficients are not computed yet. Please compute those first.')
            return None

        if ldatm == 'ck2004' and photon_weighted:
            axes = self._ck2004_intensity_axes
            table = self._ck2004_ld_photon_grid
        elif ldatm == 'phoenix' and photon_weighted:
            axes = self._phoenix_intensity_axes
            table = self._phoenix_ld_photon_grid
        elif ldatm == 'ck2004' and not photon_weighted:
            axes = self._ck2004_intensity_axes
            table = self._ck2004_ld_energy_grid
        elif ldatm == 'phoenix' and not photon_weighted:
            axes = self._phoenix_intensity_axes
            table = self._phoenix_ld_energy_grid
        else:
            print('ldatm=%s is not supported for LD interpolation.' % ldatm)
            return None

        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, abun),))
            ld_coeffs = libphoebe.interp(req, axes[0:3], table)[0]
        else:
            req = np.vstack((Teff, logg, abun)).T
            ld_coeffs = libphoebe.interp(req, axes[0:3], table).T

        nanmask = np.isnan(np.sum(ld_coeffs, axis=0))
        if not hasattr(nanmask, '__iter__'):
            nanmask = np.array((nanmask,))
        if np.any(nanmask):
            raise ValueError('Atmosphere parameters out of bounds: ldatm=%s, teff=%s, logg=%s, abun=%s' % (ldatm, req[:,0][nanmask], req[:,1][nanmask], req[:,2][nanmask]))

        if ld_func == 'linear':
            return ld_coeffs[0:1]
        elif ld_func == 'logarithmic':
            return ld_coeffs[1:3]
        elif ld_func == 'square_root':
            return ld_coeffs[3:5]
        elif ld_func == 'quadratic':
            return ld_coeffs[5:7]
        elif ld_func == 'power':
            return ld_coeffs[7:11]
        elif ld_func == 'all':
            return ld_coeffs
        else:
            print('ld_func=%s is invalid; please choose from [linear, logarithmic, square_root, quadratic, power, all].' % ld_func)
            return None

    def interpolate_extinct(self, Teff=5772., logg=4.43, abun=0.0, atm='blackbody',  extinct=0.0, Rv=3.1, photon_weighted=False):
        """
        Interpolates the passband-stored tables of extinction corrections

        Arguments
        ----------
        * `Teff` (float, optional, default=5772): effective temperature.
        * `logg` (float, optional, default=4.43): log surface gravity
        * `abun` (float, optional, default=0.0): abundance
        * `atm` (string, optional, default='blackbody'): atmosphere model.
        * `extinct` (float, optional, default=0.0)
        * `Rv` (float, optional, default=3.1)
        * `photon_weighted` (bool, optional, default=False)

        Returns
        ---------
        * extinction factor

        Raises
        --------
        * NotImplementedError if `atm` is not supported.
        """

        if atm == 'ck2004':
            if 'ck2004:ext' not in self.content:
                raise ValueError('Extinction factors are not computed yet. Please compute those first.')

            if photon_weighted:
                table = self._ck2004_extinct_photon_grid
            else:
                table = self._ck2004_extinct_energy_grid

            if not hasattr(Teff, '__iter__'):
                req = np.array(((Teff, logg, abun, extinct, Rv),))
                extinct_factor = libphoebe.interp(req, self._ck2004_extinct_axes[0:5], table)[0][0]
            else:
                extinct=extinct*np.ones(len(Teff))
                Rv=Rv*np.ones(len(Teff))
                req = np.vstack((Teff, logg, abun, extinct, Rv)).T
                extinct_factor = libphoebe.interp(req, self._ck2004_extinct_axes[0:5], table).T[0]

            nanmask = np.isnan(extinct_factor)
            if np.any(nanmask):
                raise ValueError('Atmosphere parameters out of bounds: atm=%s, extinct=%f, Rv=%f, Teff=%s, logg=%s, abun=%s' % (atm, extinct, Rv, Teff[nanmask], logg[nanmask], abun[nanmask]))

            return extinct_factor

        if atm == 'phoenix':
            if 'phoenix:ext' not in self.content:
                raise ValueError('Extinction factors are not computed yet. Please compute those first.')

            if photon_weighted:
                table = self._phoenix_extinct_photon_grid
            else:
                table = self._phoenix_extinct_energy_grid

            if not hasattr(Teff, '__iter__'):
                req = np.array(((Teff, logg, abun, extinct, Rv),))
                extinct_factor = libphoebe.interp(req, self._phoenix_extinct_axes, table)[0][0]
            else:
                extinct=extinct*np.ones_like(Teff)
                Rv=Rv*np.ones_like(Teff)
                req = np.vstack((Teff, logg, abun, extinct, Rv)).T
                extinct_factor = libphoebe.interp(req, self._phoenix_extinct_axes, table).T[0]

            nanmask = np.isnan(extinct_factor)
            if np.any(nanmask):
                raise ValueError('Atmosphere parameters out of bounds: atm=%s, extinct=%f, Rv=%f, Teff=%s, logg=%s, abun=%s' % (atm, extinct, Rv, Teff[nanmask], logg[nanmask], abun[nanmask]))

            return extinct_factor

        elif atm != 'blackbody':
            raise  NotImplementedError("atm='{}' not currently supported".format(atm))
        else :
            if 'blackbody:ext' not in self.content:
                raise ValueError('Extinction factors are not computed yet. Please compute those first.')

            if photon_weighted:
                table = self._bb_extinct_photon_grid
            else:
                table = self._bb_extinct_energy_grid

            if not hasattr(Teff, '__iter__'):
                req = np.array(((Teff, extinct, Rv),))
                extinct_factor = libphoebe.interp(req, self._bb_extinct_axes[0:3], table)[0][0]
            else:
                extinct=extinct*np.ones(len(Teff))
                Rv=Rv*np.ones(len(Teff))
                req = np.vstack((Teff, extinct, Rv)).T
                extinct_factor = libphoebe.interp(req, self._bb_extinct_axes[0:3], table).T[0]

            nanmask = np.isnan(extinct_factor)
            if np.any(nanmask):
                raise ValueError('Atmosphere parameters out of bounds: atm=%s, extinct=%f, Rv=%f, Teff=%s, logg=%s, abun=%s' % (atm, extinct, Rv, Teff[nanmask], logg[nanmask], abun[nanmask]))

            return extinct_factor

    def import_wd_atmcof(self, plfile, atmfile, wdidx, Nabun=19, Nlogg=11, Npb=25, Nints=4):
        """
        Parses WD's atmcof and reads in all Legendre polynomials for the
        given passband.

        Arguments
        -----------
        * `plfile` (string): path and filename of atmcofplanck.dat
        * `atmfile` (string): path and filename of atmcof.dat
        * `wdidx` (int): WD index of the passed passband. Starts with 1, so
            it is aligned with the enumeration in lc and dc sources.
        * `Nabun` (int, optional, default=19): number of metallicity nodes in
            atmcof.dat. For the 2003 version the number of nodes is 19.
        * `Nlogg` (int, optional, default=11): number of logg nodes in
            atmcof.dat. For the 2003 version the number of nodes is 11.
        * `Nbp` (int, optional, default=25): number of passbands in atmcof.dat.
            For the 2003 version the number of passbands is 25.
        * `Nints` (int, optional, default=4): number of temperature intervals
            (input lines) per entry. For the 2003 version the number of lines
            is 4.
        """

        if wdidx <= 0 or wdidx > Npb:
            raise ValueError('wdidx value out of bounds: 1 <= wdidx <= Npb')

        # Store the passband index for use in planckint() and atmx():
        self.extern_wd_idx = wdidx

        # Store atmcof and atmcofplanck for independent lookup:
        # FIXME: it makes no sense to store the entire table for all passbands;
        # fix this entire logic to store only a single passband information.
        self.wd_data = libphoebe.wd_readdata(_bytes(plfile), _bytes(atmfile))

        # That is all that was necessary for *_extern_planckint() and
        # *_extern_atmx() functions. However, we also want to support
        # circumventing WD subroutines and use WD tables directly. For
        # that, we need to do a bit more work.

        # Break up the table along axes and extract a single passband data:
        # atmtab = np.reshape(self.wd_data['atm_table'], (Nabun, Npb, Nlogg, Nints, -1))
        # atmtab = atmtab[:,wdidx-1,:,:,:]

        # Finally, reverse the metallicity axis because it is sorted in
        # reverse order in atmcof:
        # self.extern_wd_atmx = atmtab[::-1,:,:,:]
        self.content += ['extern_planckint:Inorm', 'extern_atmx:Inorm']

    def _log10_Inorm_extern_planckint(self, Teff):
        """
        Internal function to compute normal passband intensities using
        the external WD machinery that employs blackbody approximation.

        @Teff: effective temperature in K

        Returns: log10(Inorm)
        """

        log10_Inorm = libphoebe.wd_planckint(Teff, self.extern_wd_idx, self.wd_data["planck_table"])

        return log10_Inorm

    def _log10_Inorm_extern_atmx(self, Teff, logg, abun):
        """
        Internal function to compute normal passband intensities using
        the external WD machinery that employs model atmospheres and
        ramps.

        Arguments
        ----------
        * `Teff`: effective temperature in K
        * `logg`: surface gravity in cgs
        * `abun`: metallicity in dex, Solar=0.0

        Returns
        ----------
        * log10(Inorm)
        """

        log10_Inorm = libphoebe.wd_atmint(Teff, logg, abun, self.extern_wd_idx, self.wd_data["planck_table"], self.wd_data["atm_table"])

        return log10_Inorm

    def _log10_Inorm_ck2004(self, Teff, logg, abun, photon_weighted=False):
        #~ if not hasattr(Teff, '__iter__'):
            #~ req = np.array(((Teff, logg, abun),))
            #~ log10_Inorm = libphoebe.interp(req, self._ck2004_axes, self._ck2004_photon_grid if photon_weighted else self._ck2004_energy_grid)[0][0]
        #~ else:
        req = np.vstack((Teff, logg, abun)).T
        log10_Inorm = libphoebe.interp(req, self._ck2004_axes, self._ck2004_photon_grid if photon_weighted else self._ck2004_energy_grid).T[0]

        return log10_Inorm

    def _Inorm_ck2004(self, Teff, logg, abun, photon_weighted=False):
        req = np.vstack((Teff, logg, abun)).T
        Inorm = libphoebe.interp(req, self._ck2004_axes, self._ck2004_photon_grid if photon_weighted else self._ck2004_energy_grid).T[0]

        return 10**Inorm

    def _Inorm_phoenix(self, Teff, logg, abun, photon_weighted=False):
        req = np.vstack((Teff, logg, abun)).T
        Inorm = libphoebe.interp(req, self._phoenix_axes, self._phoenix_photon_grid if photon_weighted else self._phoenix_energy_grid).T[0]

        return 10**Inorm

    def _log10_Imu_ck2004(self, Teff, logg, abun, mu, photon_weighted=False):
        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, abun, mu),))
            log10_Imu = libphoebe.interp(req, self._ck2004_intensity_axes, self._ck2004_Imu_photon_grid if photon_weighted else self._ck2004_Imu_energy_grid)[0][0]
        else:
            req = np.vstack((Teff, logg, abun, mu)).T
            log10_Imu = libphoebe.interp(req, self._ck2004_intensity_axes, self._ck2004_Imu_photon_grid if photon_weighted else self._ck2004_Imu_energy_grid).T[0]

        return log10_Imu

    def _Imu_ck2004(self, Teff, logg, abun, mu, photon_weighted=False):
        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, abun, mu),))
            Imu = libphoebe.interp(req, self._ck2004_intensity_axes, self._ck2004_Imu_photon_grid if photon_weighted else self._ck2004_Imu_energy_grid)[0][0]
        else:
            req = np.vstack((Teff, logg, abun, mu)).T
            Imu = libphoebe.interp(req, self._ck2004_intensity_axes, self._ck2004_Imu_photon_grid if photon_weighted else self._ck2004_Imu_energy_grid).T[0]

        return 10**Imu

    def _Imu_phoenix(self, Teff, logg, abun, mu, photon_weighted=False):
        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, abun, mu),))
            Imu = libphoebe.interp(req, self._phoenix_intensity_axes, self._phoenix_Imu_photon_grid if photon_weighted else self._phoenix_Imu_energy_grid)[0][0]
        else:
            req = np.vstack((Teff, logg, abun, mu)).T
            Imu = libphoebe.interp(req, self._phoenix_intensity_axes, self._phoenix_Imu_photon_grid if photon_weighted else self._phoenix_Imu_energy_grid).T[0]

        return 10**Imu

    def Inorm(self, Teff=5772., logg=4.43, abun=0.0, atm='ck2004', ldatm='ck2004', ldint=None, ld_func='interp', ld_coeffs=None, photon_weighted=False):
        """

        Arguments
        ----------
        * `Teff`
        * `logg`
        * `abun`
        * `atm`
        * `ldatm`
        * `ldint` (string, optional, default=None): integral of the limb
            darkening function, \int_0^1 \mu L(\mu) d\mu. Its general role is to
            convert intensity to flux. In this method, however, it is only needed
            for blackbody atmospheres because they are not limb-darkened (i.e.
            the blackbody intensity is the same irrespective of \mu), so we need
            to *divide* by ldint to ascertain the correspondence between
            luminosity, effective temperature and fluxes once limb darkening
            correction is applied at flux integration time. If None, and if
            `atm=='blackbody'`, it will be computed from `ld_func` and
            `ld_coeffs`.
        * `ld_func` (string, optional, default='interp') limb darkening
            function.  One of: linear, sqrt, log, quadratic, power, interp.
        * `ld_coeffs` (list, optional, default=None): limb darkening coefficients
            for the corresponding limb darkening function, `ld_func`.
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        ----------
        * (float/array) normal intensities.


        Raises
        ----------
        * ValueError: if atmosphere parameters are out of bounds for the table.
        * NotImplementedError: if `ld_func` is not supported.
        """
        # TODO: improve docstring

        # convert scalars to vectors if necessary:
        if not hasattr(Teff, '__iter__'):
            Teff = np.array((Teff,))
        if not hasattr(logg, '__iter__'):
            logg = np.array((logg,))
        if not hasattr(abun, '__iter__'):
            abun = np.array((abun,))

        if atm == 'blackbody' and 'blackbody:Inorm' in self.content:
            if photon_weighted:
                retval = 10**self._log10_Inorm_bb_photon(Teff)
            else:
                retval = 10**self._log10_Inorm_bb_energy(Teff)
            if ldint is None:
                ldint = self.ldint(Teff, logg, abun, ldatm, ld_func, ld_coeffs, photon_weighted)
            retval /= ldint

        elif atm == 'extern_planckint' and 'extern_planckint:Inorm' in self.content:
            # -1 below is for cgs -> SI:
            retval = 10**(self._log10_Inorm_extern_planckint(Teff)-1)
            if ldint is None:
                ldint = self.ldint(Teff, logg, abun, ldatm, ld_func, ld_coeffs, photon_weighted)
            retval /= ldint

        elif atm == 'extern_atmx' and 'extern_atmx:Inorm' in self.content:
            # -1 below is for cgs -> SI:
            retval = 10**(self._log10_Inorm_extern_atmx(Teff, logg, abun)-1)

        elif atm == 'ck2004' and 'ck2004:Inorm' in self.content:
            retval = self._Inorm_ck2004(Teff, logg, abun, photon_weighted=photon_weighted)

        elif atm == 'phoenix' and 'phoenix:Inorm' in self.content:
            retval = self._Inorm_phoenix(Teff, logg, abun, photon_weighted=photon_weighted)

        else:
            raise NotImplementedError('atm={} not supported by {}:{}'.format(atm, self.pbset, self.pbname))

        nanmask = np.isnan(retval)
        if np.any(nanmask):
            raise ValueError('Atmosphere parameters out of bounds: atm=%s, ldatm=%s, Teff=%s, logg=%s, abun=%s' % (atm, ldatm, Teff[nanmask], logg[nanmask], abun[nanmask]))
        return retval

    def Imu(self, Teff=5772., logg=4.43, abun=0.0, mu=1.0, atm='ck2004', ldatm='ck2004', ldint=None, ld_func='interp', ld_coeffs=None, photon_weighted=False):
        """
        Arguments
        ----------
        * `Teff`
        * `logg`
        * `abun`
        * `atm`
        * `ldatm`
        * `ldint` (string, optional, default='ck2004'): integral of the limb
            darkening function, \int_0^1 \mu L(\mu) d\mu. Its general role is to
            convert intensity to flux. In this method, however, it is only needed
            for blackbody atmospheres because they are not limb-darkened (i.e.
            the blackbody intensity is the same irrespective of \mu), so we need
            to *divide* by ldint to ascertain the correspondence between
            luminosity, effective temperature and fluxes once limb darkening
            correction is applied at flux integration time. If None, and if
            `atm=='blackbody'`, it will be computed from `ld_func` and
            `ld_coeffs`.
        * `ld_func` (string, optional, default='interp') limb darkening
            function.  One of: linear, sqrt, log, quadratic, power, interp.
        * `ld_coeffs` (list, optional, default=None): limb darkening coefficients
            for the corresponding limb darkening function, `ld_func`.
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        ----------
        * (float/array) projected intensities.

        Raises
        ----------
        * ValueError: if atmosphere parameters are out of bounds for the table.
        * ValueError: if `ld_func='interp'` but is not supported by the
            atmosphere table.
        * NotImplementedError: if `ld_func` is not supported.
        """
        # TODO: improve docstring

        # make sure we're not suffering from rounding issues in mu:
        mu[np.isclose(mu, 1)] = 1-1e-12
        mu[np.isclose(mu, 0)] = 1e-12

        if ld_func == 'interp':
            # The 'interp' LD function works only for model atmospheres:
            if atm == 'ck2004' and 'ck2004:Imu' in self.content:
                retval = self._Imu_ck2004(Teff, logg, abun, mu, photon_weighted=photon_weighted)
                nanmask = np.isnan(retval)
                if np.any(nanmask):
                    raise ValueError('Atmosphere parameters out of bounds: Teff=%s, logg=%s, abun=%s, mu=%s' % (Teff[nanmask], logg[nanmask], abun[nanmask], mu[nanmask]))
                return retval
            elif atm == 'phoenix' and 'phoenix:Imu' in self.content:
                retval = self._Imu_phoenix(Teff, logg, abun, mu, photon_weighted=photon_weighted)
                nanmask = np.isnan(retval)
                if np.any(nanmask):
                    raise ValueError('Atmosphere parameters out of bounds: Teff=%s, logg=%s, abun=%s, mu=%s' % (Teff[nanmask], logg[nanmask], abun[nanmask], mu[nanmask]))
                return retval
            else:
                raise ValueError('atm={} not supported by {}:{} ld_func=interp'.format(atm, self.pbset, self.pbname))

        if ld_coeffs is None:
            # LD function can be passed without coefficients; in that
            # case we need to interpolate them from the tables.
            ld_coeffs = self.interpolate_ldcoeffs(Teff, logg, abun, ldatm, ld_func, photon_weighted)

        if ld_func == 'linear':
            retval = self.Inorm(Teff=Teff, logg=logg, abun=abun, atm=atm, ldatm=ldatm, ldint=ldint, ld_func=ld_func, ld_coeffs=ld_coeffs, photon_weighted=photon_weighted) * self._ldlaw_lin(mu, *ld_coeffs)
        elif ld_func == 'logarithmic':
            retval = self.Inorm(Teff=Teff, logg=logg, abun=abun, atm=atm, ldatm=ldatm, ldint=ldint, ld_func=ld_func, ld_coeffs=ld_coeffs, photon_weighted=photon_weighted) * self._ldlaw_log(mu, *ld_coeffs)
        elif ld_func == 'square_root':
            retval = self.Inorm(Teff=Teff, logg=logg, abun=abun, atm=atm, ldatm=ldatm, ldint=ldint, ld_func=ld_func, ld_coeffs=ld_coeffs, photon_weighted=photon_weighted) * self._ldlaw_sqrt(mu, *ld_coeffs)
        elif ld_func == 'quadratic':
            retval = self.Inorm(Teff=Teff, logg=logg, abun=abun, atm=atm, ldatm=ldatm, ldint=ldint, ld_func=ld_func, ld_coeffs=ld_coeffs, photon_weighted=photon_weighted) * self._ldlaw_quad(mu, *ld_coeffs)
        elif ld_func == 'power':
            retval = self.Inorm(Teff=Teff, logg=logg, abun=abun, atm=atm, ldatm=ldatm, ldint=ldint, ld_func=ld_func, ld_coeffs=ld_coeffs, photon_weighted=photon_weighted) * self._ldlaw_nonlin(mu, *ld_coeffs)
        else:
            raise NotImplementedError('ld_func={} not supported'.format(ld_func))

        nanmask = np.isnan(retval)
        if np.any(nanmask):
            raise ValueError('Atmosphere parameters out of bounds: Teff=%s, logg=%s, abun=%s, mu=%s' % (Teff[nanmask], logg[nanmask], abun[nanmask], mu[nanmask]))
        return retval

    def _ldint_ck2004(self, Teff, logg, abun, photon_weighted):
        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, abun),))
            ldint = libphoebe.interp(req, self._ck2004_axes, self._ck2004_ldint_photon_grid if photon_weighted else self._ck2004_ldint_energy_grid)[0][0]
        else:
            req = np.vstack((Teff, logg, abun)).T
            ldint = libphoebe.interp(req, self._ck2004_axes, self._ck2004_ldint_photon_grid if photon_weighted else self._ck2004_ldint_energy_grid).T[0]

        return ldint

    def _ldint_phoenix(self, Teff, logg, abun, photon_weighted):
        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, abun),))
            ldint = libphoebe.interp(req, self._phoenix_axes, self._phoenix_ldint_photon_grid if photon_weighted else self._phoenix_ldint_energy_grid)[0][0]
        else:
            req = np.vstack((Teff, logg, abun)).T
            ldint = libphoebe.interp(req, self._phoenix_axes, self._phoenix_ldint_photon_grid if photon_weighted else self._phoenix_ldint_energy_grid).T[0]

        return ldint

    def ldint(self, Teff=5772., logg=4.43, abun=0.0, ldatm='ck2004', ld_func='interp', ld_coeffs=None, photon_weighted=False):
        """
        Arguments
        ----------
        * `Teff`
        * `logg`
        * `abun`
        * `ldatm`
        * `ld_func` (string, optional, default='interp') limb darkening
            function.  One of: linear, sqrt, log, quadratic, power, interp.
        * `ld_coeffs` (list, optional, default=None): limb darkening coefficients
            for the corresponding limb darkening function, `ld_func`.
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        ----------
        * (float/array) ldint.

        Raises
        ----------
        * ValueError: if atmosphere parameters are out of bounds for the table.
        * ValueError: if `ld_func='interp'` but is not supported by the
            atmosphere table.
        * NotImplementedError: if `ld_func` is not supported.
        """
        # TODO: improve docstring

        if ld_func == 'interp':
            if ldatm == 'ck2004':
                retval = self._ldint_ck2004(Teff, logg, abun, photon_weighted=photon_weighted)
            elif ldatm == 'phoenix':
                retval = self._ldint_phoenix(Teff, logg, abun, photon_weighted=photon_weighted)
            else:
                raise ValueError('ldatm={} not supported with ld_func=interp'.format(ldatm))
            nanmask = np.isnan(retval)
            if np.any(nanmask):
                raise ValueError('Atmosphere parameters out of bounds: Teff=%s, logg=%s, abun=%s' % (Teff[nanmask], logg[nanmask], abun[nanmask]))
            return retval

        if ld_coeffs is None:
            ld_coeffs = self.interpolate_ldcoeffs(Teff, logg, abun, ldatm, ld_func, photon_weighted)

        if ld_func == 'linear':
            retval = 1-ld_coeffs[0]/3
        elif ld_func == 'logarithmic':
            retval = 1-ld_coeffs[0]/3+2.*ld_coeffs[1]/9
        elif ld_func == 'square_root':
            retval = 1-ld_coeffs[0]/3-ld_coeffs[1]/5
        elif ld_func == 'quadratic':
            retval = 1-ld_coeffs[0]/3-ld_coeffs[1]/6
        elif ld_func == 'power':
            retval = 1-ld_coeffs[0]/5-ld_coeffs[1]/3-3.*ld_coeffs[2]/7-ld_coeffs[3]/2
        else:
            raise NotImplementedError('ld_func={} not supported'.format(ld_func))

        nanmask = np.isnan(retval)
        if np.any(nanmask):
            raise ValueError('Atmosphere parameters out of bounds: Teff=%s, logg=%s, abun=%s' % (Teff[nanmask], logg[nanmask], abun[nanmask]))
        return retval

    def _bindex_ck2004(self, Teff, logg, abun, mu, atm, photon_weighted=False):
        grid = self._ck2004_boosting_photon_grid if photon_weighted else self._ck2004_boosting_energy_grid
        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, abun, mu),))
            bindex = libphoebe.interp(req, self._ck2004_intensity_axes, grid)[0][0]
        else:
            req = np.vstack((Teff, logg, abun, mu)).T
            bindex = libphoebe.interp(req, self._ck2004_intensity_axes, grid).T[0]

        return bindex

    def bindex(self, Teff=5772., logg=4.43, abun=0.0, mu=1.0, atm='ck2004', photon_weighted=False):
        """
        Arguments
        ----------
        * `Teff`
        * `logg`
        * `abun`
        * `mu`
        * `atm`
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        ----------
        * (float/array) boosting index

        Raises
        ----------
        * ValueError: if atmosphere parameters are out of bounds for the table.
        * NotImplementedError: if `atm` is not supported (not one of 'ck2004'
            or 'blackbody').
        """
        # TODO: implement phoenix boosting.

        if atm == 'ck2004':
            retval = self._bindex_ck2004(Teff, logg, abun, mu, atm, photon_weighted)
        elif atm == 'blackbody':
            retval = self._bindex_blackbody(Teff, photon_weighted=photon_weighted)
        else:
            raise NotImplementedError('atm={} not supported'.format(atm))

        nanmask = np.isnan(retval)
        if np.any(nanmask):
            raise ValueError('Atmosphere parameters out of bounds: Teff=%s, logg=%s, abun=%s' % (Teff[nanmask], logg[nanmask], abun[nanmask]))
        return retval

def _timestamp_to_dt(timestamp):
    if timestamp is None:
        return None
    elif not isinstance(timestamp, str):
        raise TypeError("timestamp not of type string")
    return datetime.strptime(timestamp, "%a %b %d %H:%M:%S %Y")

def _init_passband(fullpath, check_for_update=True):
    """
    """
    global _pbtable
    logger.info("initializing passband (headers only) at {}".format(fullpath))
    try:
        pb = Passband.load(fullpath, load_content=False)
    except:
        print("failed to load passband at {}".format(fullpath))
        raise
    passband = pb.pbset+':'+pb.pbname
    atms = list(set([c.split(':')[0] for c in pb.content]))
    atms_ld = [atm for atm in atms if '{}:ld'.format(atm) in pb.content and '{}:ldint'.format(atm) in pb.content]
    dirname = os.path.dirname(fullpath)
    if dirname == os.path.dirname(_pbdir_local):
        local = True
    elif dirname == os.path.dirname(_pbdir_global):
        local = False
    else:
        local = None
    _pbtable[passband] = {'fname': fullpath, 'content': pb.content, 'atms': atms, 'atms_ld': atms_ld, 'timestamp': pb.timestamp, 'pb': None, 'local': local}

def _init_passbands(refresh=False, query_online=True, passband_directories=None):
    """
    This function should be called only once, at import time. It
    traverses the passbands directory and builds a lookup table of
    passband names qualified as 'pbset:pbname' and corresponding files
    and atmosphere content within.
    """
    global _initialized
    global _pbtable

    if passband_directories is None:
        passband_directories = list_passband_directories()

    if isinstance(passband_directories, str):
        passband_directories = [passband_directories]

    if not _initialized or refresh:
        # load information from online passbands first so that any that are
        # available locally will override
        if query_online:
            online_passbands = list_online_passbands(full_dict=True, refresh=refresh, repeat_errors=False)
            for pb, info in online_passbands.items():
                _pbtable[pb] = {'fname': None, 'atms': info['atms'], 'atms_ld': info.get('atms_ld', ['ck2004']), 'pb': None}

        # load global passbands (in install directory) next and then local
        # (in .phoebe directory) second so that local passbands override
        # global passbands whenever there is a name conflict
        for path in passband_directories:
            for f in os.listdir(path):
                if f == 'README':
                    continue
                if ".".join(f.split('.')[1:]) not in ['fits', 'fits.gz']:
                    # ignore old passband versions
                    continue
                try:
                    _init_passband(os.path.join(path, f))
                except IOError:
                    print("PHOEBE: passband from {} failed to load, skipping".format(os.path.join(path, f)))
                    pass

        _initialized = True

def install_passband(fname, local=True):
    """
    For convenience, this function is available at the top-level as
    <phoebe.install_passband> as well as
    <phoebe.atmospheres.passbands.install_passband>.

    Install a passband from a local file.  This simply copies the file into the
    install path - but beware that clearing the installation will clear the
    passband as well.

    The local and global installation directories can be listed by calling
    <phoebe.atmospheres.passbands.list_passband_directories>.  The local
    (`local=True`) directory is generally at
    `~/.phoebe/atmospheres/tables/passbands`, and the global (`local=False`)
    directory is in the PHOEBE installation directory.

    See also:
    * <phoebe.atmospheres.passbands.uninstall_all_passbands>

    Arguments
    ----------
    * `fname` (string) the filename of the local passband.
    * `local` (bool, optional, default=True): whether to install to the local/user
        directory or the PHOEBE installation directory.  If `local=False`, you
        must have the necessary permissions to write to the installation
        directory.
    """
    pbdir = _pbdir_local if local else _pbdir_global
    shutil.copy(fname, pbdir)
    _init_passband(os.path.join(pbdir, fname))

def uninstall_passband(passband, local=True):
    """
    For convenience, this function is available at the top-level as
    <phoebe.uninstall_passband> as well as
    <phoebe.atmospheres.passband.uninstall_passband>.

    Uninstall a given passband, either globally or locally (need to call twice to
    delete both).  This is done by deleting the file corresponding to the
    entry in
    <phoebe.atmospheres.passbands.list_installed_passbands>.  If there are multiple
    files with the same `passband` name (local vs global, for example), this
    may need to be called multiple times.

    The local and global installation directories can be listed by calling
    <phoebe.atmospheres.passbands.list_passband_directories>.  The local
    (`local=True`) directory is generally at
    `~/.phoebe/atmospheres/tables/passbands`, and the global (`local=False`)
    directory is in the PHOEBE installation directory.

    See also:
    * <phoebe.atmospheres.passbands.install_passband>
    * <phoebe.atmospheres.passbands.unininstall_all_passbands>

    Arguments
    ----------
    * `passband` (string): name of the passband.  Must be one of the installed
        passbands (see <phoebe.atmospheres.passbands.list_installed_passbands>).
    * `local` (bool, optional, default=True): whether to uninstall from the local/user
        directory or the PHOEBE installation directory.  If `local=False`, you
        must have the necessary permissions to write to the installation
        directory.

    Raises
    ----------
    * `ValueError`: if `passband` not found in <phoebe.atmospheres.passbands.list_installed_passbands>
    * `ValueError`: if the entry for `passband` in <phoebe.atmospheres.passbands.list_installed_passbands>
        is not in the correct directory according to `local`.
    """
    fname = list_installed_passbands(full_dict=True).get(passband, {}).get('fname', None)
    if fname is None:
        raise ValueError("could not find entry for '{}' in list_installed_passbands()".format(passband))

    allowed_dir = _pbdir_local if local else _pbdir_local
    if os.path.dirname(fname) != os.path.dirname(allowed_dir):
        raise ValueError("entry for '{}' was not found in {} (directory for local={})".format(passband, allowed_dir, local))

    logger.warning("deleting file: {}".format(fname))
    os.remove(fname)

    # need to update the local cache for list_installed_passbands:
    _init_passbands(refresh=True)

def uninstall_all_passbands(local=True):
    """
    For convenience, this function is available at the top-level as
    <phoebe.uninstall_all_passbands> as well as
    <phoebe.atmospheres.passband.uninstall_all_passbands>.

    Uninstall all passbands, either globally or locally (need to call twice to
    delete ALL passbands).  This is done by deleting all files in the respective
    directory.

    The local and global installation directories can be listed by calling
    <phoebe.atmospheres.passbands.list_passband_directories>.  The local
    (`local=True`) directory is generally at
    `~/.phoebe/atmospheres/tables/passbands`, and the global (`local=False`)
    directory is in the PHOEBE installation directory.

    See also:
    * <phoebe.atmospheres.passbands.install_passband>
    * <phoebe.atmospheres.passbands.uninstall_passband>

    Arguments
    ----------
    * `local` (bool, optional, default=True): whether to uninstall from the local/user
        directory or the PHOEBE installation directory.  If `local=False`, you
        must have the necessary permissions to write to the installation
        directory.
    """
    pbdir = _pbdir_local if local else _pbdir_global
    for f in os.listdir(pbdir):
        pbpath = os.path.join(pbdir, f)
        logger.warning("deleting file: {}".format(pbpath))
        os.remove(pbpath)

    # need to update the local cache for list_installed_passbands:
    _init_passbands(refresh=True)

def download_passband(passband, content=None, local=True, gzipped=None):
    """
    For convenience, this function is available at the top-level as
    <phoebe.download_passband> as well as
    <phoebe.atmospheres.passbands.download_passband>.

    Download and install a given passband from
    http://tables.phoebe-project.org.

    The local and global installation directories can be listed by calling
    <phoebe.atmospheres.passbands.list_passband_directories>.  The local
    (`local=True`) directory is generally at
    `~/.phoebe/atmospheres/tables/passbands`, and the global (`local=False`)
    directory is in the PHOEBE installation directory.

    Arguments
    ----------
    * `passband` (string): name of the passband.  Must be one of the available
        passbands in the repository (see
        <phoebe.atmospheres.passbands.list_online_passbands>).
    * `content` (string or list or None, optional, default=None): content to fetch
        from the server.  Options include: 'all' (to fetch all available)
        or any of the available contents for that passband, 'ck2004' to fetch
        all contents for the 'ck2004' atmosphere, or any specific list of
        available contents.  To see available options for a given passband, see
        the 'content' entry for a given passband in the dictionary exposed by
        <phoebe.atmospheres.passbands.list_online_passbands>
        with `full_dict=True`.  If None, will respect options in
        <phoebe.set_download_passband_defaults>.
    * `local` (bool, optional, default=True): whether to install to the local/user
        directory or the PHOEBE installation directory.  If `local=False`, you
        must have the necessary permissions to write to the installation
        directory.
    * `gzipped` (bool or None, optional, default=None): whether to download a
        compressed version of the passband.  Compressed files take up less
        disk-space and less time to download, but take approximately 1 second
        to load (which will happen once per-passband per-session).  If None,
        will respect options in <phoebe.set_download_passband_defaults>.

    Raises
    --------
    * ValueError: if the value of `passband` is not one of
        <phoebe.atmospheres.passbands.list_online_passbands>.
    * IOError: if internet connection fails.
    """
    if passband not in list_online_passbands(repeat_errors=False):
        raise ValueError("passband '{}' not available".format(passband))

    if content is None:
        content = conf.download_passband_defaults.get('content', 'all')
        logger.info("adopting content={} from phoebe.get_download_passband_defaults()".format(content))
    if gzipped is None:
        gzipped = conf.download_passband_defaults.get('gzipped', False)
        logger.info("adopting gzipped={} from phoebe.get_download_passband_defaults()".format(gzipped))

    pbdir = _pbdir_local if local else _pbdir_global

    if isinstance(content, str):
        content_str = content
    elif isinstance(content, list) or isinstance(content, tuple):
        content_str = ",".join(content)
    else:
        raise TypeError("content must be of type string or list")

    if list_installed_passbands(full_dict=True).get(passband, {}).get('local', None) == local:
        logger.warning("passband '{}' already exists with local={}... removing".format(passband, local))
        uninstall_passband(passband, local=local)

    passband_fname_local = os.path.join(pbdir, passband.lower().replace(':', '_')+".fits")
    if gzipped:
        passband_fname_local += '.gz'
    url = '{}/pbs/{}/{}?phoebe_version={}&gzipped={}'.format(_url_tables_server, passband, content_str, phoebe_version, gzipped)
    logger.info("downloading from {} and installing to {}...".format(url, passband_fname_local))
    try:
        urlretrieve(url, passband_fname_local)
    except IOError as e:
        raise IOError("unable to download {} passband - check connection.  Original error: {}".format(passband, e))
    else:
        _init_passband(passband_fname_local)

def list_passband_online_history(passband, since_installed=True):
    """
    For convenience, this function is available at the top-level as
    <phoebe.list_passband_online_history> as well as
    <phoebe.atmospheres.passbands.list_passband_online_history>.

    Access the full changelog for the online version of a passband.

    See also:
    * <phoebe.atmospheres.passbands.update_passband_available>
    * <phoebe.atmospheres.passbands.list_all_update_passbands_available>

    Arguments
    ------------
    * `passband` (string): name of the passband
    * `since_installed` (bool, optional, default=True): whether to filter
        the changelog entries to only those since the timestamp of the installed
        version.

    Returns
    ----------
    * (dict): dictionary with timestamps as keys and messages and values.
    """
    if passband not in list_online_passbands(repeat_errors=False):
        raise ValueError("'{}' passband not availabe online".format(passband))

    url = '{}/pbs/history/{}?phoebe_version={}'.format(_url_tables_server, passband, phoebe_version)

    try:
        resp = urlopen(url, timeout=3)
    except Exception as err:
        msg = "connection to online passbands at {} could not be established.  Check your internet connection or try again later.  If the problem persists and you're using a Mac, you may need to update openssl (see http://phoebe-project.org/help/faq).".format(_url_tables_server)
        msg += " Original error from urlopen: {} {}".format(err.__class__.__name__, str(err))

        logger.warning(msg)
        return {str(time.ctime()): "could not retrieve history entries"}
    else:
        try:
            all_history = json.loads(resp.read().decode('utf-8'), object_pairs_hook=parse_json).get('passband_history', {}).get(passband, {})
        except Exception as err:
            msg = "Parsing response from online passbands at {} failed.".format(_url_tables_server)
            msg += " Original error from json.loads: {} {}".format(err.__class__.__name__, str(err))

            logger.warning(msg)
            return {str(time.ctime()): "could not parse history entries"}

        if since_installed:
            installed_timestamp = _timestamp_to_dt(_pbtable.get(passband, {}).get('timestamp', None))
            return {k:v for k,v in all_history.items() if installed_timestamp < _timestamp_to_dt(k)} if installed_timestamp is not None else all_history
        else:
            return all_history

def update_passband_available(passband, history_dict=False):
    """
    For convenience, this function is available at the top-level as
    <phoebe.update_passband_available> as well as
    <phoebe.atmospheres.passbands.update_passband_available>.

    Check if a newer version of a given passband is available from the online repository.
    Note that this does not check to see if more atmosphere tables are available
    but were not fetched.  To see that, compare the output of
    <phoebe.atmospheres.passbands.list_installed_passbands> and
    <phoebe.atmospheres.passbands.list_online_passbands> with `full_dict=True`.

    If a new version is available, you can update by calling <phoebe.atmospheres.passbands.download_passband>.

    See also:
    * <phoebe.atmospheres.passbands.list_all_update_passbands_available>
    * <phoebe.atmospheres.passbands.list_passband_online_history>
    * <phoebe.atmospheres.passbands.download_passband>
    * <phoebe.atmospheres.passbands.update_all_passbands>

    Arguments
    -----------
    * `passband` (string): name of the passband
    * `history_dict` (boolean, optional, default=False): expose the changelog
        of the version online since the timestamp in the installed version.
        See also: <phoebe.atmospheres.passbands.list_passband_online_history>.

    Returns
    -----------
    * (bool or dict): whether a newer version is available.  Boolean if
        `history_dict=False`.  Dictionary of changelog entries since the current
        version with timestamps as keys and messages as values if `history_dict=True`
        (will be empty if no updates available).
    """
    def _return(passband, updates_available):
        if updates_available:
            if history_dict:
                return list_passband_online_history(passband, since_installed=True)
            else:
                return True
        else:
            if history_dict:
                return {}
            else:
                return False

    if passband not in list_online_passbands(repeat_errors=False):
        logger.warning("{} not available in online passbands".format(passband))
        return _return(passband, False)

    online_timestamp = _online_passbands.get(passband, {}).get('timestamp', None)
    installed_timestamp = _pbtable.get(passband, {}).get('timestamp', None)

    if online_timestamp is None:
        return _return(passband, False)

    elif installed_timestamp is None:
        if online_timestamp is not None:
            return _return(passband, True)

    elif online_timestamp is None:
        return _return(passband, False)

    else:
        try:
            installed_timestamp_dt = _timestamp_to_dt(installed_timestamp)
            online_timestamp_dt = _timestamp_to_dt(online_timestamp)
        except Exception as err:
            msg = "failed to convert passband timestamps, so cannot determine if updates are available.  To disable online passbands entirely, set the environment variable PHOEBE_ENABLE_ONLINE_PASSBANDS=FALSE.  Check tables.phoebe-project.org manually for updates.  Original error: {}".format(err)
            print("ERROR: {}".format(msg))
            logger.error(msg)
            return _return(passband, False)
        else:
            if installed_timestamp_dt < online_timestamp_dt:
                return _return(passband, True)

    return _return(passband, False)

def list_all_update_passbands_available(history_dict=False):
    """
    For convenience, this function is available at the top-level as
    <phoebe.list_all_update_passbands_available> as well as
    <phoebe.atmospheres.passbands.list_all_update_passbands_available>.

    See also:
    * <phoebe.atmospheres.passbands.update_passband_available>
    * <phoebe.atmospheres.passbands.list_passband_online_history>
    * <phoebe.atmospheres.passbands.download_passband>
    * <phoebe.atmospheres.passbands.update_all_passbands>

    Arguments
    -----------
    * `history_dict` (boolean, optional, default=False): for each item in
        the returned list, expose the changelog.  See also:
        <phoebe.atmospheres.passbands.list_passband_online_history>.

    Returns
    ----------
    * (list of strings or dict): list of passbands with newer versions available
        online.  If `history_dict=False`, this will be a list of strings,
        where each item is the passband name.  If `history_dict=True` this will
        be a dictionary where the keys are the passband names and the values
        are the changelog dictionary (see <phoebe.atmospheres.passbands.list_passband_online_history>).
    """
    if history_dict:
        return {p: update_passband_available(p, history_dict=True) for p in list_installed_passbands() if update_passband_available(p)}
    else:
        return [p for p in list_installed_passbands() if update_passband_available(p)]

def update_passband(passband, local=True, content=None, gzipped=None):
    """
    For convenience, this function is available at the top-level as
    <phoebe.update_passbands> as well as
    <phoebe.atmospheres.passbands.update_passband>.

    Download and install updates for a single passband from
    http://tables.phoebe-project.org, retrieving
    the same content as in the installed passband.

    This will install into the directory dictated by `local`, regardless of the
    location of the original file.  `local`=True passbands always override
    `local=False`.

    The local and global installation directories can be listed by calling
    <phoebe.atmospheres.passbands.list_passband_directories>.  The local
    (`local=True`) directory is generally at
    `~/.phoebe/atmospheres/tables/passbands`, and the global (`local=False`)
    directory is in the PHOEBE installation directory.

    See also:
    * <phoebe.atmospheres.passbands.download_passband>
    * <phoebe.atmospheres.passbands.update_passband_available>
    * <phoebe.atmospheres.passbands.list_all_update_passbands_available>
    * <phoebe.atmospheres.passbands.update_all_passbands>


    Arguments
    ----------
    * `passband` (string): passband to update
    * `local` (bool, optional, default=True): whether to install to the local/user
        directory or the PHOEBE installation directory.  If `local=False`, you
        must have the necessary permissions to write to the installation
        directory.
    * `content` (string or list, optional, default=None): content to request
        when downloading the passband, in addition to any content in the existing
        installed passband, if applicable.
        Options include: None (request the same contents as the installed version),
        'all' (to update with all available content),
        'ck2004' to require all contents for the 'ck2004' atmosphere, or any specific list of
        available contents.  To see available options for a given passband, see
        the 'content' entry for a given passband in the dictionary exposed by
        <phoebe.atmospheres.passbands.list_online_passbands>
        with `full_dict=True`.
    * `gzipped` (bool or None, optional, default=None): whether to download a
        compressed version of the passband.  Compressed files take up less
        disk-space and less time to download, but take approximately 1 second
        to load (which will happen once per-passband per-session).  If None,
        will respect options in <phoebe.set_download_passband_defaults>.

    Raises
    --------
    * IOError: if internet connection fails.
    """
    installed_content = list_installed_passbands(full_dict=True).get(passband, {}).get('content', [])
    if content is None:
        content = installed_content
    elif isinstance(content, str):
        if content != 'all':
            content = list(set(installed_content + [content]))
    elif isinstance(content, list):
        content = list(set(installed_content + content))
    else:
        raise TypeError("content must be of type list, string, or None")

    # TODO: if same timestamp online and local, only download new content and merge
    download_passband(passband, content=content, local=local, gzipped=gzipped)

def update_all_passbands(local=True, content=None):
    """
    For convenience, this function is available at the top-level as
    <phoebe.update_all_passbands> as well as
    <phoebe.atmospheres.passbands.update_all_passbands>.

    Download and install updates for all passbands from
    http://tables.phoebe-project.org, retrieving
    the same content as in the installed passbands.

    This will install into the directory dictated by `local`, regardless of the
    location of the original file.  `local`=True passbands always override
    `local=False`.

    The local and global installation directories can be listed by calling
    <phoebe.atmospheres.passbands.list_passband_directories>.  The local
    (`local=True`) directory is generally at
    `~/.phoebe/atmospheres/tables/passbands`, and the global (`local=False`)
    directory is in the PHOEBE installation directory.

    See also
    * <phoebe.atmospheres.passbands.list_all_update_passbands_available>
    * <phoebe.atmospheres.passbands.update_passband>
    * <phoebe.atmospheres.passbands.update_passband_available>


    Arguments
    ----------
    * `local` (bool, optional, default=True): whether to install to the local/user
        directory or the PHOEBE installation directory.  If `local=False`, you
        must have the necessary permissions to write to the installation
        directory.
    * `content` (string or list, optional, default=None): content to request
        when downloading the passband, in addition to any content in the existing
        installed passband, if applicable.
        Options include: None (request the same contents as the installed version),
        'all' (to update with all available content),
        'ck2004' to require all contents for the 'ck2004' atmosphere, or any specific list of
        available contents.  To see available options for a given passband, see
        the 'content' entry for a given passband in the dictionary exposed by
        <phoebe.atmospheres.passbands.list_online_passbands>
        with `full_dict=True`.

    Raises
    --------
    * IOError: if internet connection fails.
    """
    for passband in list_all_update_passbands_available():
        update_passband(passband, local=local, content=content)

def list_passband_directories():
    """
    For convenience, this function is available at the top-level as
    <phoebe.list_passband_directories> as well as
    <phoebe.atmospheres.passbands.list_passband_directories>.

    List the global and local passband installation directories (in that order).

    The local and global installation directories can be listed by calling
    <phoebe.atmospheres.passbands.list_passband_directories>.  The local
    (`local=True`) directory is generally at
    `~/.phoebe/atmospheres/tables/passbands`, and the global (`local=False`)
    directory is in the PHOEBE installation directory.

    Returns
    --------
    * (list of strings): global and local passband installation directories.
    """
    return [p for p in [_pbdir_global, _pbdir_local, _pbdir_env] if p is not None]

def list_passbands(refresh=False, full_dict=False, skip_keys=[]):
    """
    For convenience, this function is available at the top-level as
    <phoebe.list_passbands> as well as
    <phoebe.atmospheres.passbands.list_passbands>.

    List all available passbands, both installed and available online.

    This is just a combination of
    <phoebe.atmospheres.passbands.list_installed_passbands> and
    <phoebe.atmospheres.passbands.list_online_passbands>.

    Arguments
    ---------
    * `refresh` (bool, optional, default=False): whether to refresh the list
        of fallback on cached values.  Passing `refresh=True` should only
        be necessary if new passbands have been installed or added to the
        online repository since importing PHOEBE.
    * `full_dict` (bool, optional, default=False): whether to return the full
        dictionary of information about each passband or just the list
        of names.
    * `skip_keys` (list, optional, default=[]): keys to exclude from the returned
        dictionary.  Only applicable if `full_dict` is True.

    Returns
    --------
    * (list of strings or dictionary, depending on `full_dict`)
    """
    if full_dict:
        d = list_online_passbands(refresh, True, skip_keys=skip_keys, repeat_errors=False)
        for k in d.keys():
            if 'installed' not in skip_keys:
                d[k]['installed'] = False
        # installed passband always overrides online
        for k,v in list_installed_passbands(refresh, True, skip_keys=skip_keys).items():
            d[k] = v
            if 'installed' not in skip_keys:
                d[k]['installed'] = True
        return d
    else:
        return list(set(list_installed_passbands(refresh) + list_online_passbands(refresh, repeat_errors=False)))

def list_installed_passbands(refresh=False, full_dict=False, skip_keys=[]):
    """
    For convenience, this function is available at the top-level as
    <phoebe.list_installed_passbands> as well as
    <phoebe.atmospheres.passbands.list_installed_passbands>.

    List all installed passbands, both in the local and global directories.

    See also:
    * <phoebe.atmospheres.passbands.list_passband_directories>

    Arguments
    ---------
    * `refresh` (bool, optional, default=False): whether to refresh the list
        of fallback on cached values.  Passing `refresh=True` should only
        be necessary if new passbands have been installed or added to the
        online repository since importing PHOEBE.
    * `full_dict` (bool, optional, default=False): whether to return the full
        dictionary of information about each passband or just the list
        of names.
    * `skip_keys` (list, optional, default=[]): keys to exclude from the returned
        dictionary.  Only applicable if `full_dict` is True.

    Returns
    --------
    * (list of strings or dictionary, depending on `full_dict`)
    """

    if refresh:
        _init_passbands(True)

    if full_dict:
        return {k:_dict_without_keys(v, skip_keys) for k,v in _pbtable.items() if v['fname'] is not None}
    else:
        return [k for k,v in _pbtable.items() if v['fname'] is not None]

def list_online_passbands(refresh=False, full_dict=False, skip_keys=[], repeat_errors=True):
    """
    For convenience, this function is available at the top-level as
    <phoebe.list_online_passbands> as well as
    <phoebe.atmospheres.passbands.list_online_passbands>.

    List all passbands available for download from
    http://tables.phoebe-project.org.

    Arguments
    ---------
    * `refresh` (bool, optional, default=False): whether to refresh the list
        of fallback on cached values.  Passing `refresh=True` should only
        be necessary if new passbands have been installed or added to the
        online repository since importing PHOEBE.
    * `full_dict` (bool, optional, default=False): whether to return the full
        dictionary of information about each passband or just the list
        of names.
    * `skip_keys` (list, optional, default=[]): keys to exclude from the returned
        dictionary.  Only applicable if `full_dict` is True.
    * `repeat_errors` (bool, optional, default=True): whether to continue to show
        errors if online passbands are unavailable.  (Internally this is passed
        as False so that the error message does not spam the log, but defaults
        to True so if calling manually the error message is shown).

    Returns
    --------
    * (list of strings or dictionary, depending on `full_dict`)
    """
    global _online_passbands
    global _online_passband_failedtries
    if os.getenv('PHOEBE_ENABLE_ONLINE_PASSBANDS', 'TRUE').upper() == 'TRUE' and (len(_online_passbands.keys())==0 or refresh):
        if _online_passband_failedtries >= 3 and not refresh:
            if ((_online_passband_failedtries >= 3 and repeat_errors) or (_online_passband_failedtries==3)):
                msg = "Online passbands unavailable (reached max tries).  Pass refresh=True to force another attempt or repeat_errors=False to avoid showing this message."
                logger.warning(msg)
            _online_passband_failedtries += 1
        else:
            url = '{}/pbs/list?phoebe_version={}'.format(_url_tables_server, phoebe_version)

            try:
                resp = urlopen(url, timeout=3)
            except Exception as err:
                _online_passband_failedtries += 1
                msg = "Connection to online passbands at {} could not be established.  Check your internet connection or try again later (can manually call phoebe.list_online_passbands(refresh=True) to retry).  If the problem persists and you're using a Mac, you may need to update openssl (see http://phoebe-project.org/help/faq).".format(_url_tables_server, _online_passband_failedtries)
                msg += " Original error from urlopen: {} {}".format(err.__class__.__name__, str(err))

                logger.warning("(Attempt {} of 3): ".format(_online_passband_failedtries)+msg)
                # also print in case logger hasn't been initialized yet
                if _online_passband_failedtries == 1:
                    print(msg)

                if _online_passbands is not None:
                    if full_dict:
                        return {k:_dict_without_keys(v, skip_keys) for k,v in _online_passbands.items()}
                    else:
                        return list(_online_passbands.keys())
                else:
                    if full_dict:
                        return {}
                    else:
                        return []
            else:
                try:
                    _online_passbands = json.loads(resp.read().decode('utf-8'), object_pairs_hook=parse_json)['passbands_list']
                except Exception as err:
                    _online_passband_failedtries += 1
                    msg = "Parsing response from online passbands at {} failed.".format(_url_tables_server)
                    msg += " Original error from json.loads: {} {}".format(err.__class__.__name__, str(err))

                    logger.warning("(Attempt {} of 3): ".format(_online_passband_failedtries)+msg)
                    # also print in case logger hasn't been initialized yet
                    if _online_passband_failedtries == 1:
                        print(msg)

                    if _online_passbands is not None:
                        if full_dict:
                            return {k:_dict_without_keys(v, skip_keys) for k,v in _online_passbands.items()}
                        else:
                            return list(_online_passbands.keys())
                    else:
                        if full_dict:
                            return {}
                        else:
                            return []

    if full_dict:
        return {k:_dict_without_keys(v, skip_keys) for k,v in _online_passbands.items()}
    else:
        return list(_online_passbands.keys())

def get_passband(passband, content=None, reload=False, update_if_necessary=False,
                 download_local=True, download_gzipped=None):
    """
    For convenience, this function is available at the top-level as
    <phoebe.get_passband> as well as
    <phoebe.atmospheres.passbands.get_passband>.

    Access a passband object by name.  If the passband isn't installed, it
    will be downloaded and installed locally.  If the installed passband does
    not have the necessary tables to match `content` then an attempt will be
    made to download the necessary additional tables from
    http://tables.phoebe-project.org
    as long as the timestamps match the local version.  If the online version
    includes other version updates, then an error will be
    raised suggesting to call <phoebe.atmospheres.passbands.update_passband>
    unless `update_if_necessary` is passed as True, in which case the update
    will automatically be downloaded and installed.

    See also:
    * <phoebe.atmospheres.passbands.list_installed_passbands>
    * <phoebe.atmospheres.passbands.download_passband>
    * <phoebe.atmospheres.passbands.list_passband_directories>

    Arguments
    -----------
    * `passband` (string): name of the passband.  Must be one of the available
        passbands in the repository (see
        <phoebe.atmospheres.passbands.list_online_passbands>).
    * `content` (string or list, optional, default=None): content to require
        to retrieve from a local passband... otherwise will download and install
        the passband by passing `content` to
        <phoebe.atmospheres.passbands.download_passband>.
        Options include: None (to accept the content in the local version,
        but to respect options in <phoebe.set_download_passband_defaults>
        if no installed version exists), 'all' (to require and fetch all
        available content),
        'ck2004' to require and fetch
        all contents for the 'ck2004' atmosphere only (for example), or any specific list of
        available contents.  To see available options for a given passband, see
        the 'content' entry for a given passband in the dictionary exposed by
        <phoebe.atmospheres.passbands.list_online_passbands>
        with `full_dict=True`.
    * `reload` (bool, optional, default=False): force reloading from the
        local file even if a copy of the passband exists in memory.
    * `update_if_necessary` (bool, optional, default=False): if a local version
        exists, but does not contain the necessary requirements according to
        `content`, and the online version has a different timestamp than the
        installed version, then an error will be raised unless `update_if_necessary`
        is set to True.
    * `download_local` (bool, optional, default=True): Only applicable if the
        passband has to be downloaded from the server.  Whether to install to the local/user
        directory or the PHOEBE installation directory.  If `local=False`, you
        must have the necessary permissions to write to the installation
        directory.
    * `download_gzipped` (bool or None, optional, default=None): Only applicable if
        the passband has to be downloaded from the server.  Whether to download a
        compressed version of the passband.  Compressed files take up less
        disk-space and less time to download, but take approximately 1 second
        to load (which will happen once per-passband per-session).  If None,
        will respect options in <phoebe.set_download_passband_defaults>.

    Returns
    -----------
    * the passband object

    Raises
    --------
    * ValueError: if the passband cannot be found installed or online.
    * ValueError: if the passband cannot be found installed and online passbands
        are unavailable (due to server being down or online passbands disabled
        by environment variable).
    * IOError: if needing to download the passband but the connection fails.
    """
    global _pbtable

    if passband in list_installed_passbands():
        # then we need to make sure all the required content are met in the local version
        content_installed = _pbtable[passband]['content']
        timestamp_installed = _pbtable[passband]['timestamp']
        online_content = list_online_passbands(full_dict=True, repeat_errors=False).get(passband, {}).get('content', [])

        if content == 'all':
            content = online_content
        elif content is not None:
            if isinstance(content, str):
                content = [content]
            # need to account for mixed atm/table content = ['ck2004', 'blackbody:Inorm']
            content_expanded = []
            for c in content:
                if ':' in c:
                    content_expanded.append(c)
                else:
                    content_expanded += [oc for oc in online_content if oc.split(':')[0]==c]
            # and lastly remove any duplicated from expanding content = ['ck2004', 'ck2004:ld']
            content = list(set(content_expanded))

        if content is not None and not np.all([c in content_installed for c in content]):
            # then we can update without prompting if the timestamps match
            timestamp_online = list_online_passbands(full_dict=True, repeat_errors=False).get(passband, {}).get('timestamp', None)
            if timestamp_online is not None and (update_if_necessary or timestamp_installed == timestamp_online):
                download_passband(passband, content=content, local=download_local, gzipped=download_gzipped)
            else:
                # TODO: ValueError may not be the right choice here...
                raise ValueError("installed version of {} passband does not meet content={} requirements, but online version has a different timestamp.  Call get_passband with update_if_necessary=True or call update_passband to force updating to the newer version.")

        else:
            # then we will just retrieve the local version and return it
            pass
    elif os.getenv('PHOEBE_ENABLE_ONLINE_PASSBANDS', 'TRUE').upper() == 'TRUE':
        # then we need to download, if available online
        if passband in list_online_passbands(repeat_errors=False):
            download_passband(passband, content=content, local=download_local, gzipped=download_gzipped)
        else:
            raise ValueError("passband: {} not found. Try one of: {} (local) or {} (available for download)".format(passband, list_installed_passbands(), list_online_passbands(repeat_errors=False)))

    else:
        raise ValueError("passband {} not installed locally and online passbands is disabled.".format(passband))

    if reload or _pbtable.get(passband, {}).get('pb', None) is None:
        logger.info("loading {} passband from {} (including all tables)".format(passband, _pbtable[passband]['fname']))
        pb = Passband.load(_pbtable[passband]['fname'], load_content=True)
        _pbtable[passband]['pb'] = pb

    return _pbtable[passband]['pb']

def Inorm_bol_bb(Teff=5772., logg=4.43, abun=0.0, atm='blackbody', photon_weighted=False):
    """
    Computes normal bolometric intensity using the Stefan-Boltzmann law,
    Inorm_bol_bb = 1/\pi \sigma T^4. If photon-weighted intensity is
    requested, Inorm_bol_bb is multiplied by a conversion factor that
    comes from integrating lambda/hc P(lambda) over all lambda.

    Input parameters mimick the <phoebe.atmospheres.passbands.Passband.Inorm>
    method for calling convenience.

    Arguments
    ------------
    * `Teff` (float/array, optional, default=5772):  value or array of effective
        temperatures.
    * `logg` (float/array, optional, default=4.43): IGNORED, for class
        compatibility only.
    * `abun` (float/array, optional, default=0.0): IGNORED, for class
        compatibility only.
    * `atm` (string, optional, default='blackbody'): atmosphere model, must be
        `'blackbody'`, otherwise exception is raised.
    * `photon_weighted` (bool, optional, default=False): photon-weighted or
        energy-weighted mode.

    Returns
    ---------
    * (float/array) float or array (depending on input types) of normal
        bolometric blackbody intensities.

    Raises
    --------
    * ValueError: if `atm` is anything other than `'blackbody'`.
    """
    # TODO: the docs say errors will be raised if photon_weighted is not False
    # but this doesn't seem to be the case.

    if atm != 'blackbody':
        raise ValueError('atmosphere must be set to blackbody for Inorm_bol_bb.')

    if photon_weighted:
        factor = 2.6814126821264836e22/Teff
    else:
        factor = 1.0

    # convert scalars to vectors if necessary:
    if not hasattr(Teff, '__iter__'):
        Teff = np.array((Teff,))

    return factor * sigma_sb.value * Teff**4 / np.pi

if __name__ == '__main__':
    # This will generate bolometric and Johnson V passband files. Note that
    # extinction for the bolometric band cannot be computed because it falls
    # off the extinction formula validity range in wavelength, and shouldn't
    # be computed anyway because it is only used for reflection purposes.

    pb = Passband(
        ptf='bolometric.ptf',
        pbset='Bolometric',
        pbname='900-40000',
        effwl=1.955e-6,
        wlunits=u.m,
        calibrated=True,
        reference='Flat response to simulate bolometric throughput',
        version=2.0,
        comments=''
    )

    pb.compute_blackbody_response()

    pb.compute_ck2004_response(path='tables/ck2004fits', verbose=True)
    pb.compute_ck2004_intensities(path='tables/ck2004fits', verbose=True)
    pb.compute_ck2004_ldcoeffs()
    pb.compute_ck2004_ldints()

    pb.compute_phoenix_response(path='tables/phoenix', verbose=True)
    pb.compute_phoenix_intensities(path='tables/phoenix', verbose=True)
    pb.compute_phoenix_ldcoeffs()
    pb.compute_phoenix_ldints()

    pb.save('bolometric.fits')

    pb = Passband(
        ptf='johnson_v.ptf',
        pbset='Johnson',
        pbname='V',
        effwl=5500.,
        wlunits=u.AA,
        calibrated=True,
        reference='Maiz Apellaniz (2006), AJ 131, 1184',
        version=2.0,
        comments=''
    )

    pb.compute_blackbody_response()
    pb.compute_bb_reddening(verbose=True)

    pb.compute_ck2004_response(path='tables/ck2004', verbose=True)
    pb.compute_ck2004_intensities(path='tables/ck2004', verbose=True)
    pb.compute_ck2004_ldcoeffs()
    pb.compute_ck2004_ldints()
    pb.compute_ck2004_reddening(path='tables/ck2004', verbose=True)

    pb.compute_phoenix_response(path='tables/phoenix', verbose=True)
    pb.compute_phoenix_intensities(path='tables/phoenix', verbose=True)
    pb.compute_phoenix_ldcoeffs()
    pb.compute_phoenix_ldints()
    pb.compute_phoenix_reddening(path='tables/phoenix', verbose=True)

    pb.import_wd_atmcof('tables/wd/atmcofplanck.dat', 'tables/wd/atmcof.dat', 7)

    pb.save('johnson_v.fits')
