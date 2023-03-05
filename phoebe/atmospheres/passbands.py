from phoebe import __version__ as phoebe_version
from phoebe import conf, mpi
from phoebe.utils import _bytes
from phoebe.dependencies import ndpolator

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
from scipy.spatial import cKDTree
from datetime import datetime
import libphoebe
import os
import sys
import glob
import shutil
import json
import time
import re

# NOTE: python3 only
from urllib.request import urlopen, urlretrieve

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
_supported_atms = ['blackbody', 'ck2004', 'phoenix', 'tmap', 'extern_atmx', 'extern_planckint']

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


def _dict_without_keys(d, skip_keys=[]):
    return {k: v for k, v in d.items() if k not in skip_keys}

def blending_factor(d, func='sigmoid', scale=15, offset=0.5):
    """
    Computes the amount of blending for coordinate `d`.

    This auxiliary function returns a factor between 0 and 1 that is used for
    blending a model atmosphere into blackbody atmosphere as the atmosphere
    values fall off the grid. By default the function uses a sigmoid to
    compute the factor, where a sigmoid is defined as:

    f(d) = 1 - (1 + e^{-tau (d-Delta)})^{-1},

    where tau is scaling and Delta is offset.

    Arguments
    ---------
    * `d` (float or array): distance or distances from the grid
    * `func` (string, optional, default='sigmoid'):
        type of blending function; it can be 'linear' or 'sigmoid'
    * `scale` (float, optional, default=15):
        if `func`='sigmoid', `scale` is the scaling for the sigmoid
    * `offset` (float, optional, default=0.5):
        if `func`='sigmoid', `offset` is the zero-point between 0 and 1.

    Returns
    -------
    * (float) blending factor between 0 and 1
    """

    rv = np.zeros_like(d)
    if func == 'linear':
        rv[d <= 1] = 1-d[d <= 1]
    elif func == 'sigmoid':
        rv[d <= 1] = 1-(1+np.exp(-scale*(d[d <= 1]-offset)))**-1
    else:
        print('function `%s` not supported.' % func)
        return None
    rv[d < 0] = 1
    return rv

def raise_out_of_bounds(nanvals, atm=None, ldatm=None, intens_weighting=None):
    value_error = 'atmosphere parameters out of bounds: '
    if atm is not None:
        value_error += f'atm={atm} '
    if ldatm is not None:
        value_error += f'ldatm={ldatm} '
    if intens_weighting is not None:
        value_error += f'intens_weighting={intens_weighting} '
    value_error += f'values={nanvals}'
    raise ValueError(value_error)


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
        pb.compute_blackbody_intensities()
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
            2-column file with wavelength in `wlunits` and transmission in
            arbitrary units.
        * `pbset` (string, optional, default='Johnson'): name of the passband
            set (i.e. Johnson).
        * `pbname` (string, optional, default='V'): name of the passband name
            (i.e. V).
        * `effwl` (float, optional, default=5500.0): effective wavelength in
            `wlunits`.
        * `wlunits` (unit, optional, default=u.AA): wavelength units from
            astropy.units used in `ptf` and `effwl`.
        * `calibrated` (bool, optional, default=False): True if transmission is
            in true fractional light, False if it is in relative proportions.
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

        self.h = h.value
        self.c = c.value
        self.k = k_B.value

        if from_file:
            return

        if "'" in pbset or '"' in pbset:
            raise ValueError("pbset cannot contain quotation marks")
        if "'" in pbname or '"' in pbname:
            raise ValueError("pbname cannot contain quotation marks")

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
        self.wl_oversampling = oversampling
        self.wl = np.linspace(self.ptf_table['wl'][0], self.ptf_table['wl'][-1], self.wl_oversampling*len(self.ptf_table['wl']))

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
        self.history = []

        # If any comments are passed, add them to history:
        if comments != '':
            self.add_to_history(comments)

        # Initialize passband tables:
        self.atm_axes = dict()
        self.ext_axes = dict()
        self.atm_energy_grid = dict()
        self.atm_photon_grid = dict()
        self.ld_energy_grid = dict()
        self.ld_photon_grid = dict()
        self.ldint_energy_grid = dict()
        self.ldint_photon_grid = dict()
        self.ext_energy_grid = dict()
        self.ext_photon_grid = dict()
        self.nntree = dict()
        self.indices = dict()
        self.ics = dict()
        self.blending_region = dict()
        self.mapper = dict()
        self.ndp = dict()

    def __repr__(self):
        return f'<Passband: {self.pbset}:{self.pbname}>'

    def __str__(self):
        # old passband files do not have versions embedded, that is why we have to do this:
        if not hasattr(self, 'version') or self.version is None:
            self.version = 1.0
        return f'Passband: {self.pbset}:{self.pbname}\nVersion:  {self.version:1.1f}\nProvides: {self.content}\nHistory:  {self.history}'

    def add_to_history(self, comment):
        """
        Adds a comment to the passband file header.

        Parameters
        ----------
        * `comment` (string, required): comment to be added to the passband header.
        """

        if not isinstance(comment, str):
            raise ValueError('passband header comments must be strings.')
        
        self.history.append(f'{time.ctime()}: {comment}')

    def on_updated_ptf(self, ptf, wlunits=u.AA, oversampling=1, ptf_order=3):
        """
        When passband transmission function is updated, this function updates
        all related meta-fields in the passband structure. It does *not* update
        any tables, only the header information.
        """

        ptf_table = np.loadtxt(ptf).T
        ptf_table[0] = ptf_table[0]*wlunits.to(u.m)
        self.ptf_table = {'wl': np.array(ptf_table[0]), 'fl': np.array(ptf_table[1])}

        self.wl_oversampling = oversampling
        self.wl = np.linspace(self.ptf_table['wl'][0], self.ptf_table['wl'][-1], self.wl_oversampling*len(self.ptf_table['wl']))

        self.ptf_order = ptf_order
        self.ptf_func = interpolate.splrep(self.ptf_table['wl'], self.ptf_table['fl'], s=0, k=ptf_order)
        self.ptf = lambda wl: interpolate.splev(wl, self.ptf_func)
        self.ptf_area = interpolate.splint(self.wl[0], self.wl[-1], self.ptf_func, 0)

        # Spline fit to the photon-weighted passband transmission function table:
        self.ptf_photon_func = interpolate.splrep(self.ptf_table['wl'], self.ptf_table['fl']*self.ptf_table['wl'], s=0, k=ptf_order)
        self.ptf_photon = lambda wl: interpolate.splev(wl, self.ptf_photon_func)
        self.ptf_photon_area = interpolate.splint(self.wl[0], self.wl[-1], self.ptf_photon_func, 0)

        return

    def save(self, archive, overwrite=True, update_timestamp=True, export_inorm_tables=False, history_entry=''):
        """
        Saves the passband file in the fits format.

        Arguments
        ----------
        * `archive` (string): filename of the saved file
        * `overwrite` (bool, optional, default=True): whether to overwrite an
            existing file with the same filename as provided in `archive`
        * `update_timestamp` (bool, optional, default=True): whether to update
            the stored timestamp with the current time.
        * `export_inorm_tables` (bool, optional, default=False): Inorm tables
            have been deprecated since phoebe 2.4; for backwards compatibility
            we still may need to export them to fits files so that pre-2.4
            versions can use the same passband files.
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
        header['WLOVSMPL'] = self.wl_oversampling
        header['VERSION'] = self.version
        header['COMMENTS'] = self.comments
        header['REFERENC'] = self.reference
        header['PTFORDER'] = self.ptf_order
        header['PTFEAREA'] = self.ptf_area
        header['PTFPAREA'] = self.ptf_photon_area

        header['CONTENT'] = str(self.content)

        # Add all existing history entries:
        for entry in self.history:
            header['history'] = entry

        # Append any new history entry:
        if history_entry:
            self.history.append(f'{timestamp}: {history_entry}')
            header['history'] = self.history[-1]

        if 'extern_planckint:Inorm' in self.content or 'extern_atmx:Inorm' in self.content:
            header['WD_IDX'] = self.extern_wd_idx

        data = []

        # Header:
        primary_hdu = fits.PrimaryHDU(header=header)
        data.append(primary_hdu)

        # Tables:
        data.append(fits.table_to_hdu(Table(self.ptf_table, meta={'extname': 'PTFTABLE'})))

        if 'blackbody:Inorm' in self.content:
            bb_func = Table({
                'teff': self._bb_func_energy[0],
                'logi_e': self._bb_func_energy[1],
                'logi_p': self._bb_func_photon[1]},
                meta={'extname': 'BB_FUNC'
            })
            data.append(fits.table_to_hdu(bb_func))

        if 'blackbody:ext' in self.content:
            data.append(fits.table_to_hdu(Table({'teff': self._bb_extinct_axes[0]}, meta={'extname': 'BB_TEFFS'})))
            data.append(fits.table_to_hdu(Table({'ebv': self._bb_extinct_axes[1]}, meta={'extname': 'BB_EBVS'})))
            data.append(fits.table_to_hdu(Table({'rv': self._bb_extinct_axes[2]}, meta={'extname': 'BB_RVS'})))

        if 'ck2004:Imu' in self.content:
            ck_teffs, ck_loggs, ck_abuns, ck_mus = self.atm_axes['ck2004']
            ck_teffs, ck_loggs, ck_abuns, ck_mus = self.atm_axes['ck2004']
            data.append(fits.table_to_hdu(Table({'teff': ck_teffs}, meta={'extname': 'CK_TEFFS'})))
            data.append(fits.table_to_hdu(Table({'logg': ck_loggs}, meta={'extname': 'CK_LOGGS'})))
            data.append(fits.table_to_hdu(Table({'abun': ck_abuns}, meta={'extname': 'CK_ABUNS'})))
            data.append(fits.table_to_hdu(Table({'mu': ck_mus}, meta={'extname': 'CK_MUS'})))

        if 'ck2004:ext' in self.content:
            ck_teffs, ck_loggs, ck_abuns, ck_ebvs, ck_rvs = self.ext_axes['ck2004']
            data.append(fits.table_to_hdu(Table({'ebv': ck_ebvs}, meta={'extname': 'CK_EBVS'})))
            data.append(fits.table_to_hdu(Table({'rv': ck_rvs}, meta={'extname': 'CK_RVS'})))

        if 'phoenix:Imu' in self.content:
            ph_teffs, ph_loggs, ph_abuns, ph_mus = self.atm_axes['phoenix']
            ph_teffs, ph_loggs, ph_abuns, ph_mus = self.atm_axes['phoenix']
            data.append(fits.table_to_hdu(Table({'teff': ph_teffs}, meta={'extname': 'PH_TEFFS'})))
            data.append(fits.table_to_hdu(Table({'logg': ph_loggs}, meta={'extname': 'PH_LOGGS'})))
            data.append(fits.table_to_hdu(Table({'abun': ph_abuns}, meta={'extname': 'PH_ABUNS'})))
            data.append(fits.table_to_hdu(Table({'mu': ph_mus}, meta={'extname': 'PH_MUS'})))

        if 'phoenix:ext' in self.content:
            ph_teffs, ph_loggs, ph_abuns, ph_ebvs, ph_rvs = self.ext_axes['phoenix']
            data.append(fits.table_to_hdu(Table({'ebv': ph_ebvs}, meta={'extname': 'PH_EBVS'})))
            data.append(fits.table_to_hdu(Table({'rv': ph_rvs}, meta={'extname': 'PH_RVS'})))

        if 'tmap:Imu' in self.content:
            tm_teffs, tm_loggs, tm_abuns, tm_mus = self.atm_axes['tmap']
            tm_teffs, tm_loggs, tm_abuns, tm_mus = self.atm_axes['tmap']
            data.append(fits.table_to_hdu(Table({'teff': tm_teffs}, meta={'extname': 'TM_TEFFS'})))
            data.append(fits.table_to_hdu(Table({'logg': tm_loggs}, meta={'extname': 'TM_LOGGS'})))
            data.append(fits.table_to_hdu(Table({'abun': tm_abuns}, meta={'extname': 'TM_ABUNS'})))
            data.append(fits.table_to_hdu(Table({'mu': tm_mus}, meta={'extname': 'TM_MUS'})))

        if 'tmap:ext' in self.content:
            tm_teffs, tm_loggs, tm_abuns, tm_ebvs, tm_rvs = self.ext_axes['tmap']
            data.append(fits.table_to_hdu(Table({'ebv': tm_ebvs}, meta={'extname': 'TM_EBVS'})))
            data.append(fits.table_to_hdu(Table({'rv': tm_rvs}, meta={'extname': 'TM_RVS'})))

        # Data:
        if 'blackbody:ext' in self.content:
            data.append(fits.ImageHDU(self._bb_extinct_energy_grid, name='BBEGRID'))
            data.append(fits.ImageHDU(self._bb_extinct_photon_grid, name='BBPGRID'))

        if 'ck2004:Imu' in self.content:
            data.append(fits.ImageHDU(self.atm_energy_grid['ck2004'], name='CKFEGRID'))
            data.append(fits.ImageHDU(self.atm_photon_grid['ck2004'], name='CKFPGRID'))

            if export_inorm_tables:
                data.append(fits.ImageHDU(self.atm_energy_grid['ck2004'][..., -1, :], name='CKNEGRID'))
                data.append(fits.ImageHDU(self.atm_photon_grid['ck2004'][..., -1, :], name='CKNPGRID'))

        if 'ck2004:ld' in self.content:
            data.append(fits.ImageHDU(self.ld_energy_grid['ck2004'], name='CKLEGRID'))
            data.append(fits.ImageHDU(self.ld_photon_grid['ck2004'], name='CKLPGRID'))

        if 'ck2004:ldint' in self.content:
            data.append(fits.ImageHDU(self.ldint_energy_grid['ck2004'], name='CKIEGRID'))
            data.append(fits.ImageHDU(self.ldint_photon_grid['ck2004'], name='CKIPGRID'))

        if 'ck2004:ext' in self.content:
            data.append(fits.ImageHDU(self.ext_energy_grid['ck2004'], name='CKXEGRID'))
            data.append(fits.ImageHDU(self.ext_photon_grid['ck2004'], name='CKXPGRID'))

        if 'phoenix:Imu' in self.content:
            data.append(fits.ImageHDU(self.atm_energy_grid['phoenix'], name='PHFEGRID'))
            data.append(fits.ImageHDU(self.atm_photon_grid['phoenix'], name='PHFPGRID'))

            if export_inorm_tables:
                data.append(fits.ImageHDU(self.atm_energy_grid['phoenix'][..., -1, :], name='PHNEGRID'))
                data.append(fits.ImageHDU(self.atm_photon_grid['phoenix'][..., -1, :], name='PHNPGRID'))

        if 'phoenix:ld' in self.content:
            data.append(fits.ImageHDU(self.ld_energy_grid['phoenix'], name='PHLEGRID'))
            data.append(fits.ImageHDU(self.ld_photon_grid['phoenix'], name='PHLPGRID'))

        if 'phoenix:ldint' in self.content:
            data.append(fits.ImageHDU(self.ldint_energy_grid['phoenix'], name='PHIEGRID'))
            data.append(fits.ImageHDU(self.ldint_photon_grid['phoenix'], name='PHIPGRID'))

        if 'phoenix:ext' in self.content:
            data.append(fits.ImageHDU(self.ext_energy_grid['phoenix'], name='PHXEGRID'))
            data.append(fits.ImageHDU(self.ext_photon_grid['phoenix'], name='PHXPGRID'))

        if 'tmap:Imu' in self.content:
            data.append(fits.ImageHDU(self.atm_energy_grid['tmap'], name='TMFEGRID'))
            data.append(fits.ImageHDU(self.atm_photon_grid['tmap'], name='TMFPGRID'))

            if export_inorm_tables:
                data.append(fits.ImageHDU(self.atm_energy_grid['tmap'][..., -1, :], name='TMNEGRID'))
                data.append(fits.ImageHDU(self.atm_photon_grid['tmap'][..., -1, :], name='TMNPGRID'))

        if 'tmap:ld' in self.content:
            data.append(fits.ImageHDU(self.ld_energy_grid['tmap'], name='TMLEGRID'))
            data.append(fits.ImageHDU(self.ld_photon_grid['tmap'], name='TMLPGRID'))

        if 'tmap:ldint' in self.content:
            data.append(fits.ImageHDU(self.ldint_energy_grid['tmap'], name='TMIEGRID'))
            data.append(fits.ImageHDU(self.ldint_photon_grid['tmap'], name='TMIPGRID'))

        if 'tmap:ext' in self.content:
            data.append(fits.ImageHDU(self.ext_energy_grid['tmap'], name='TMXEGRID'))
            data.append(fits.ImageHDU(self.ext_photon_grid['tmap'], name='TMXPGRID'))

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
            self.wl_oversampling = header.get('wlovsmpl', 1)
            self.comments = header['comments']
            self.reference = header['referenc']
            self.ptf_order = header['ptforder']
            self.ptf_area = header['ptfearea']
            self.ptf_photon_area = header['ptfparea']

            self.content = eval(header['content'], {'__builtins__':None}, {})
            self.history = list(header.get('history', ''))

            # try:
            #     history = ''.join([l.ljust(72) if '-END-' not in l else l for l in header['HISTORY']]).split('-END-')
            # except KeyError:
            #     history = []

            # self.history = {h.split(': ')[0]: ': '.join(h.split(': ')[1:]) for h in history if len(h.split(': ')) > 1}

            self.atm_axes = dict()
            self.ext_axes = dict()
            self.atm_energy_grid = dict()
            self.atm_photon_grid = dict()
            self.ld_energy_grid = dict()
            self.ld_photon_grid = dict()
            self.ldint_energy_grid = dict()
            self.ldint_photon_grid = dict()
            self.ext_energy_grid = dict()
            self.ext_photon_grid = dict()
            self.nntree = dict()
            self.indices = dict()
            self.ics = dict()
            self.blending_region = dict()
            self.mapper = dict()
            self.ndp = dict()

            self.ptf_table = hdul['ptftable'].data
            self.wl = np.linspace(self.ptf_table['wl'][0], self.ptf_table['wl'][-1], int(self.wl_oversampling*len(self.ptf_table['wl'])))

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

                if 'ck2004:Imu' in self.content:
                    self.atm_axes['ck2004'] = (np.array(list(hdul['ck_teffs'].data['teff'])), np.array(list(hdul['ck_loggs'].data['logg'])), np.array(list(hdul['ck_abuns'].data['abun'])), np.array(list(hdul['ck_mus'].data['mu'])))
                    self.atm_energy_grid['ck2004'] = hdul['ckfegrid'].data
                    self.atm_photon_grid['ck2004'] = hdul['ckfpgrid'].data

                    # Rebuild the table of non-null indices for the nearest neighbor lookup:
                    self.nntree['ck2004'], self.indices['ck2004'] = ndpolator.kdtree(self.atm_axes['ck2004'][:-1], self.atm_photon_grid['ck2004'][...,-1,:])

                    # Rebuild blending map:
                    self.blending_region['ck2004'] = ((750, 10000), (0.5, 0.5), (0.5, 0.5))
                    self.mapper['ck2004'] = lambda v: ndpolator.map_to_cube(v, self.atm_axes['ck2004'][:-1], self.blending_region['ck2004'])

                    # Rebuild the table of inferior corners for extrapolation:
                    raxes = self.atm_axes['ck2004'][:-1]
                    subgrid = self.atm_photon_grid['ck2004'][...,-1,:]
                    self.ics['ck2004'] = np.array([(i, j, k) for i in range(0, len(raxes[0])-1) for j in range(0, len(raxes[1])-1) for k in range(0, len(raxes[2])-1) if ~np.any(np.isnan(subgrid[i:i+2,j:j+2,k:k+2]))])

                    self.ndp['imu@photon@ck2004'] = ndpolator.Ndpolator(self.atm_axes['ck2004'], self.atm_photon_grid['ck2004'])
                    self.ndp['imu@energy@ck2004'] = ndpolator.Ndpolator(self.atm_axes['ck2004'], self.atm_energy_grid['ck2004'])

                if 'ck2004:ld' in self.content:
                    self.ld_energy_grid['ck2004'] = hdul['cklegrid'].data
                    self.ld_photon_grid['ck2004'] = hdul['cklpgrid'].data

                if 'ck2004:ldint' in self.content:
                    self.ldint_energy_grid['ck2004'] = hdul['ckiegrid'].data
                    self.ldint_photon_grid['ck2004'] = hdul['ckipgrid'].data

                if 'ck2004:ext' in self.content:
                    self.ext_axes['ck2004'] = (np.array(list(hdul['ck_teffs'].data['teff'])), np.array(list(hdul['ck_loggs'].data['logg'])), np.array(list(hdul['ck_abuns'].data['abun'])), np.array(list(hdul['ck_ebvs'].data['ebv'])), np.array(list(hdul['ck_rvs'].data['rv'])))
                    self.ext_energy_grid['ck2004'] = hdul['ckxegrid'].data
                    self.ext_photon_grid['ck2004'] = hdul['ckxpgrid'].data

                if 'phoenix:Imu' in self.content:
                    self.atm_axes['phoenix'] = (np.array(list(hdul['ph_teffs'].data['teff'])), np.array(list(hdul['ph_loggs'].data['logg'])), np.array(list(hdul['ph_abuns'].data['abun'])), np.array(list(hdul['ph_mus'].data['mu'])))
                    self.atm_energy_grid['phoenix'] = hdul['phfegrid'].data
                    self.atm_photon_grid['phoenix'] = hdul['phfpgrid'].data

                    # Rebuild the table of non-null indices for the nearest neighbor lookup:
                    self.nntree['phoenix'], self.indices['phoenix'] = ndpolator.kdtree(self.atm_axes['phoenix'][:-1], self.atm_photon_grid['phoenix'][...,-1,:])

                    # Rebuild blending map:
                    self.blending_region['phoenix'] = ((750, 2000), (0.5, 0.5), (0.5, 0.5))
                    self.mapper['phoenix'] = lambda v: ndpolator.map_to_cube(v, self.atm_axes['phoenix'][:-1], self.blending_region['phoenix'])

                    # Rebuild the table of inferior corners for extrapolation:
                    raxes = self.atm_axes['phoenix'][:-1]
                    subgrid = self.atm_photon_grid['phoenix'][...,-1,:]
                    self.ics['phoenix'] = np.array([(i, j, k) for i in range(0,len(raxes[0])-1) for j in range(0,len(raxes[1])-1) for k in range(0,len(raxes[2])-1) if ~np.any(np.isnan(subgrid[i:i+2,j:j+2,k:k+2]))])

                if 'phoenix:ld' in self.content:
                    self.ld_energy_grid['phoenix'] = hdul['phlegrid'].data
                    self.ld_photon_grid['phoenix'] = hdul['phlpgrid'].data

                if 'phoenix:ldint' in self.content:
                    self.ldint_energy_grid['phoenix'] = hdul['phiegrid'].data
                    self.ldint_photon_grid['phoenix'] = hdul['phipgrid'].data

                if 'phoenix:ext' in self.content:
                    self.ext_axes['phoenix'] = (np.array(list(hdul['ph_teffs'].data['teff'])), np.array(list(hdul['ph_loggs'].data['logg'])), np.array(list(hdul['ph_abuns'].data['abun'])), np.array(list(hdul['ph_ebvs'].data['ebv'])), np.array(list(hdul['ph_rvs'].data['rv'])))
                    self.ext_energy_grid['phoenix'] = hdul['phxegrid'].data
                    self.ext_photon_grid['phoenix'] = hdul['phxpgrid'].data

                if 'tmap:Imu' in self.content:
                    self.atm_axes['tmap'] = (np.array(list(hdul['tm_teffs'].data['teff'])), np.array(list(hdul['tm_loggs'].data['logg'])), np.array(list(hdul['tm_abuns'].data['abun'])), np.array(list(hdul['tm_mus'].data['mu'])))
                    self.atm_energy_grid['tmap'] = hdul['tmfegrid'].data
                    self.atm_photon_grid['tmap'] = hdul['tmfpgrid'].data

                    # Rebuild the table of non-null indices for the nearest neighbor lookup:
                    self.nntree['tmap'], self.indices['tmap'] = ndpolator.kdtree(self.atm_axes['tmap'][:-1], self.atm_photon_grid['tmap'][...,-1,:])

                    # Rebuild blending map:
                    self.blending_region['tmap'] = ((10000, 10000), (0.5, 0.5), (0.25, 0.25))
                    self.mapper['tmap'] = lambda v: ndpolator.map_to_cube(v, self.atm_axes['tmap'][:-1], self.blending_region['tmap'])

                    # Rebuild the table of inferior corners for extrapolation:
                    raxes = self.atm_axes['tmap'][:-1]
                    subgrid = self.atm_photon_grid['tmap'][...,-1,:]
                    self.ics['tmap'] = np.array([(i, j, k) for i in range(0,len(raxes[0])-1) for j in range(0,len(raxes[1])-1) for k in range(0,len(raxes[2])-1) if ~np.any(np.isnan(subgrid[i:i+2,j:j+2,k:k+2]))])

                if 'tmap:ld' in self.content:
                    self.ld_energy_grid['tmap'] = hdul['tmlegrid'].data
                    self.ld_photon_grid['tmap'] = hdul['tmlpgrid'].data

                if 'tmap:ldint' in self.content:
                    self.ldint_energy_grid['tmap'] = hdul['tmiegrid'].data
                    self.ldint_photon_grid['tmap'] = hdul['tmipgrid'].data

                if 'tmap:ext' in self.content:
                    self.ext_axes['tmap'] = (np.array(list(hdul['tm_teffs'].data['teff'])), np.array(list(hdul['tm_loggs'].data['logg'])), np.array(list(hdul['tm_abuns'].data['abun'])), np.array(list(hdul['tm_ebvs'].data['ebv'])), np.array(list(hdul['tm_rvs'].data['rv'])))
                    self.ext_energy_grid['tmap'] = hdul['tmxegrid'].data
                    self.ext_photon_grid['tmap'] = hdul['tmxpgrid'].data

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

    def compute_blackbody_intensities(self, teffs=None, include_extinction=False, rvs=None, ebvs=None, verbose=False):
        """
        Computes blackbody intensity interpolation functions/tables.

        Intensities are computed across the passed range of effective
        temperatures. If `teffs=None`, the function falls back onto the
        default temperature range, ~316K to ~501kK. It does this for two
        regimes, energy-weighted and photon-weighted. It then fits a cubic
        spline to the log(I)-Teff values and exports the interpolation
        functions _log10_Inorm_bb_energy and _log10_Inorm_bb_photon.

        If `include_extinction=True`, the function also computes the mean
        interstellar extinction tables. The tables contain correction factors
        rather than intensities; to get extincted intensities, correction
        factors need to be multiplied by the non-extincted intensities.
        Extinction table axes are passed through `rvs` and `ebvs` arrays; if
        `None`, defaults are used, namely `rvs=linspace(2, 6, 16)` and
        `ebvs=linspace(0, 3, 30)`.

        The computation is vectorized. The function takes the wavelength range
        from the passband transmission function and it computes a grid of
        Planck functions for all passed `teffs`. It then multiplies that grid
        by the passband transmission function and integrates the entire grid.
        For extinction, the function first adopts extinction functions a(x)
        and b(x) from Gordon et al. (2014), applies it to the table of Planck
        functions and then repeats the above process.

        Arguments
        ----------
        * `teffs` (array, optional, default=None): an array of effective
          temperatures. If None, a default array from ~300K to ~500000K with
          97 steps is used. The default array is uniform in log10 scale.
        * `include_extinction` (boolean, optional, default=False): should the
          extinction tables be computed as well. The mean effect of reddening
          (a weighted average) on a passband uses the Gordon et al. (2009,
          2014) prescription of extinction.
        * `rvs` (array, optional, default=None): a custom array of extinction
          factor Rv values. Rv is defined at Av / E(B-V) where Av is the
          visual extinction in magnitudes. If None, the default linspace(2, 6,
          16) is used.
        * `ebvs` (array, optional, default=None): a custom array of color
          excess E(B-V) values. If None, the default linspace(0, 3, 30) is
          used.
        * `verbose` (bool, optional, default=False): set to True to display
           progress in the terminal.
        """

        if verbose:
            print(f"Computing blackbody specific passband intensities for {self.pbset}:{self.pbname} {'with' if include_extinction else 'without'} extinction.")

        if teffs is None:
            log10teffs = np.linspace(2.5, 5.7, 97)  # this corresponds to the 316K-501187K range.
            teffs = 10**log10teffs

        wls = self.wl.reshape(-1, 1)

        # Planck functions:
        pfs = 2*self.h*self.c*self.c/wls**5*1./(np.exp(self.h*self.c/(self.k*wls@teffs.reshape(1, -1)))-1)  # (47, 97)

        self.atm_axes['blackbody'] = (np.unique(teffs),)

        # Passband-weighted Planck functions:
        pbpfs_energy = self.ptf(wls).reshape(-1, 1)*pfs  # (47, 97)
        pbints_energy = np.log10(np.trapz(pbpfs_energy, self.wl, axis=0)/self.ptf_area)
        self._bb_func_energy = interpolate.splrep(teffs, pbints_energy, s=0)
        self._log10_Inorm_bb_energy = lambda teff: interpolate.splev(teff, self._bb_func_energy)

        pbpfs_photon = wls*self.ptf(wls).reshape(-1, 1)*pfs  # (47, 97)
        pbints_photon = np.log10(np.trapz(pbpfs_photon, self.wl, axis=0)/self.ptf_photon_area)
        self._bb_func_photon = interpolate.splrep(teffs, pbints_photon, s=0)
        self._log10_Inorm_bb_photon = lambda teff: interpolate.splev(teff, self._bb_func_photon)

        if 'blackbody:Inorm' not in self.content:
            self.content.append('blackbody:Inorm')

        if include_extinction:
            if rvs is None:
                rvs = np.linspace(2., 6., 16)
            if ebvs is None:
                ebvs = np.linspace(0., 3., 30)

            self.ext_axes['blackbody'] = (np.unique(teffs), np.unique(rvs), np.unique(ebvs))

            ebv_column = np.tile(ebvs, len(rvs))
            rvebv_column = ebv_column*np.repeat(rvs, len(ebvs))
            ext_pars = np.vstack((rvebv_column, ebv_column))
            ext_func = libphoebe.gordon_extinction(wls)  # (47, 2)
            ext = ext_func @ ext_pars  # (47, 480)
            flux_fracs = 10**(-0.4*ext)  # (47, 480)
            integrand_energy = (pbpfs_energy[:, None, :]*flux_fracs[:, :, None]).T  # ~25ms  (97, 480, 47)
            integrand_photon = (pbpfs_photon[:, None, :]*flux_fracs[:, :, None]).T  # ~25ms  (97, 480, 47)
            # integrand = np.einsum('ji,jk->ikj', pbpfs, flux_fracs)  # ~30ms

            extincted_intensities_energy = np.trapz(integrand_energy, self.wl, axis=2)  # (97, 480)
            extincted_intensities_photon = np.trapz(integrand_photon, self.wl, axis=2)  # (97, 480)

            non_extincted_intensities_energy = np.trapz(pbpfs_energy, self.wl, axis=0)[:,None]  # (97, 1)
            non_extincted_intensities_photon = np.trapz(pbpfs_photon, self.wl, axis=0)[:,None]  # (97, 1)

            self.ext_energy_grid['blackbody'] = (extincted_intensities_energy/non_extincted_intensities_energy).reshape(len(teffs), len(rvs), len(ebvs), 1)  # (97, 16, 30, 1)
            self.ext_photon_grid['blackbody'] = (extincted_intensities_photon/non_extincted_intensities_photon).reshape(len(teffs), len(rvs), len(ebvs), 1)  # (97, 16, 30, 1)

            if 'blackbody:ext' not in self.content:
                self.content.append('blackbody:ext')

            if verbose:
                print('')

    def parse_atm_datafiles(self, atm, path):
        """
        Provides rules for parsing atmosphere fits files containing data.

        Arguments
        ----------
        * `atm` (string): model atmosphere name
        * `path` (string): relative or absolute path to data files

        Returns
        -------
        * `models` (ndarray): all non-null combinations of teffs/loggs/abuns
        * `teffs` (array): axis of all unique effective temperatures
        * `loggs` (array): axis of all unique surface gravities
        * `abuns` (array): axis of all unique abundances
        * `mus` (array): axis of all unique specific angles
        * `wls` (array): spectral energy distribution wavelengths
        * `brs` (array of tuples): blending regions for the lower and upper
          axis boundaries; for example, (500, 5000) would correspond to the
          blending region of 500 on the lower boundary and 5000 on the upper
          boundary.
        * `units` (float): conversion units from model atmosphere intensity
          units to W/m^3.
        """

        models = glob.glob(path+'/*fits')
        nmodels = len(models)
        teffs, loggs, abuns = np.empty(nmodels), np.empty(nmodels), np.empty(nmodels)

        if atm == 'ck2004':
            mus = np.array([0., 0.001, 0.002, 0.003, 0.005, 0.01 , 0.015, 0.02 , 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])
            wls = np.arange(900., 39999.501, 0.5)/1e10  # AA -> m
            for i, model in enumerate(models):
                relative_filename = model[model.rfind('/')+1:] # get relative pathname
                teffs[i] = float(relative_filename[1:6])
                loggs[i] = float(relative_filename[7:9])/10
                abuns[i] = float(relative_filename[10:12])/10 * (-1 if relative_filename[9] == 'M' else 1)
            brs = ((750, 10000), (0.5, 0.5), (0.5, 0.5))
            units = 1e7  # erg/s/cm^2/A -> W/m^3
        elif atm == 'phoenix':
            mus = np.array([0., 0.001, 0.002, 0.003, 0.005, 0.01 , 0.015, 0.02 , 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])
            wls = np.arange(500., 26000.)/1e10  # AA -> m
            for i, model in enumerate(models):
                relative_filename = model[model.rfind('/')+1:] # get relative pathname
                teffs[i] = float(relative_filename[1:6])
                loggs[i] = float(relative_filename[7:11])
                abuns[i] = float(relative_filename[12:16])
            brs = ((500, 1000), (0.5, 0.5), (0.5, 0.5))
            units = 1  # W/m^3
        elif atm == 'tmap':
            mus = np.array([0., 0.00136799, 0.00719419, 0.01761889, 0.03254691, 0.05183939, 0.07531619, 0.10275816, 0.13390887, 0.16847785, 0.20614219, 0.24655013, 0.28932435, 0.33406564, 0.38035639, 0.42776398, 0.47584619, 0.52415388, 0.57223605, 0.6196437, 0.66593427, 0.71067559, 0.75344991, 0.79385786, 0.83152216, 0.86609102, 0.89724188, 0.92468378, 0.9481606,  0.96745302, 0.98238112, 0.99280576, 0.99863193, 1.])
            wls = np.load(path+'/wavelengths.npy') # in meters
            for i, model in enumerate(models):
                pars = re.split('[TGA.]+', model[model.rfind('/')+1:])
                teffs[i] = float(pars[1])
                loggs[i] = float(pars[2])/100
                abuns[i] = float(pars[3])/100
            brs = ((10000, 10000), (0.5, 0.5), (0.25, 0.25))
            units = 1  # W/m^3
        else:
            raise ValueError(f'atm={atm} is not supported.')

        return models, teffs, loggs, abuns, mus, wls, brs, units

    def compute_intensities(self, atm, path, impute=False, include_extinction=False, rvs=None, ebvs=None, verbose=True):
        """
        Computes direction-dependent passband intensities using the passed `atm`
        model atmospheres.

        Arguments
        ----------
        * `atm` (string): name of the model atmosphere
        * `path` (string): path to the directory with SEDs in FITS format.
        * `impute` (boolean, optional, default=False): should NaN values within
            the grid be imputed.
        * `include_extinction` (boolean, optional, default=False): should the
            extinction tables be computed as well. The mean effect of reddening
            (a weighted average) on a passband uses the Gordon et al. (2009,
            2014) prescription of extinction.
        * `rvs` (array, optional, default=None): a custom array of extinction
          factor Rv values. Rv is defined at Av / E(B-V) where Av is the visual
          extinction in magnitudes. If None, the default linspace(2, 6, 16) is
          used.
        * `ebvs` (array, optional, default=None): a custom array of color excess
          E(B-V) values. If None, the default linspace(0, 3, 30) is used.
        * `verbose` (bool, optional, default=True): set to True to display
            progress in the terminal.
        """

        if verbose:
            print(f"Computing {atm} specific passband intensities for {self.pbset}:{self.pbname} {'with' if include_extinction else 'without'} extinction.")

        models, teffs, loggs, abuns, mus, wls, brs, units = self.parse_atm_datafiles(atm, path)
        nmodels = len(models)

        ints_energy, ints_photon = np.empty(nmodels*len(mus)), np.empty(nmodels*len(mus))

        if include_extinction:
            if rvs is None:
                rvs = np.linspace(2., 6., 16)
            if ebvs is None:
                ebvs = np.linspace(0., 3., 30)

            ebv_list = np.tile(np.repeat(ebvs, len(rvs)), nmodels)
            rv_list = np.tile(rvs, nmodels*len(ebvs))

            ext_matrix = np.rollaxis(np.array([np.split(rv_list*ebv_list, nmodels), np.split(ebv_list, nmodels)]), 1)
            ext_matrix = np.ascontiguousarray(ext_matrix)

            ext_energy, ext_photon = np.empty((nmodels, len(rvs)*len(ebvs))), np.empty((nmodels, len(rvs)*len(ebvs)))

        keep = (wls >= self.ptf_table['wl'][0]) & (wls <= self.ptf_table['wl'][-1])
        wls = wls[keep]

        for i, model in enumerate(models):
            with fits.open(model) as hdu:
                seds = hdu[0].data*units  # must be in in W/m^3

                # trim intensities to the passband limits:
                seds = seds[:,keep]

                pbints_energy = self.ptf(wls)*seds
                fluxes_energy = np.trapz(pbints_energy, wls)

                pbints_photon = wls*pbints_energy
                fluxes_photon = np.trapz(pbints_photon, wls)

                # work around log10(flux(mu=0)=0) = -inf:
                fluxes_energy[mus < 1e-12] = fluxes_photon[mus < 1e-12] = 1

                ints_energy[i*len(mus):(i+1)*len(mus)] = np.log10(fluxes_energy/self.ptf_area)         # energy-weighted intensity
                ints_photon[i*len(mus):(i+1)*len(mus)] = np.log10(fluxes_photon/self.ptf_photon_area)  # photon-weighted intensity

                if include_extinction:
                    ext_lambda = np.matmul(libphoebe.gordon_extinction(wls), ext_matrix[i])
                    flux_frac = np.exp(-0.9210340371976184*ext_lambda)  #10**(-0.4*ext_lambda)
                    # print(f'{pbints_energy.shape}, {fluxes_energy.shape}, {ext_energy.shape}, {flux_frac.shape}')
                    ext_energy[i], ext_photon[i] = np.dot([pbints_energy[-1]/fluxes_energy[-1], pbints_photon[-1]/fluxes_photon[-1]], flux_frac)

                if verbose:
                    sys.stdout.write('\r' + '%0.0f%% done.' % (100*float(i+1)/len(models)))
                    sys.stdout.flush()

            if verbose:
                print('')

        # for cmi, cmu in enumerate(mus):
        #     fl = intensities[cmi,:]

        #     # make a log-scale copy for boosting and fit a Legendre
        #     # polynomial to the Imu envelope by way of sigma clipping;
        #     # then compute a Legendre series derivative to get the
        #     # boosting index; we only take positive fluxes to keep the
        #     # log well defined.

        #     lnwl = np.log(wl[fl > 0])
        #     lnfl = np.log(fl[fl > 0]) + 5*lnwl

        #     # First Legendre fit to the data:
        #     envelope = np.polynomial.legendre.legfit(lnwl, lnfl, 5)
        #     continuum = np.polynomial.legendre.legval(lnwl, envelope)
        #     diff = lnfl-continuum
        #     sigma = np.std(diff)
        #     clipped = (diff > -sigma)

        #     # Sigma clip to get the continuum:
        #     while True:
        #         Npts = clipped.sum()
        #         envelope = np.polynomial.legendre.legfit(lnwl[clipped], lnfl[clipped], 5)
        #         continuum = np.polynomial.legendre.legval(lnwl, envelope)
        #         diff = lnfl-continuum

        #         # clipping will sometimes unclip already clipped points
        #         # because the fit is slightly different, which can lead
        #         # to infinite loops. To prevent that, we never allow
        #         # clipped points to be resurrected, which is achieved
        #         # by the following bitwise condition (array comparison):
        #         clipped = clipped & (diff > -sigma)

        #         if clipped.sum() == Npts:
        #             break

        #     derivative = np.polynomial.legendre.legder(envelope, 1)
        #     boosting_index = np.polynomial.legendre.legval(lnwl, derivative)

        #     # calculate energy (E) and photon (P) weighted fluxes and
        #     # their integrals.

        #     # calculate mean boosting coefficient and use it to get
        #     # boosting factors for energy (E) and photon (P) weighted
        #     # fluxes.

        #     boostE = (flE[fl > 0]*boosting_index).sum()/flEint
        #     boostP = (flP[fl > 0]*boosting_index).sum()/flPint
        #     boostingE[i] = boostE
        #     boostingP[i] = boostP

        self.atm_axes[atm] = (np.unique(teffs), np.unique(loggs), np.unique(abuns), np.unique(mus))
        self.atm_energy_grid[atm] = np.full(shape=[len(axis) for axis in self.atm_axes[atm]]+[1], fill_value=np.nan)
        self.atm_photon_grid[atm] = np.full(shape=[len(axis) for axis in self.atm_axes[atm]]+[1], fill_value=np.nan)
        # self._ck2004_boosting_energy_grid = np.nan*np.ones((len(self.atm_axes['ck2004'][0]), len(self.atm_axes['ck2004'][1]), len(self.atm_axes['ck2004'][2]), len(self.atm_axes['ck2004'][3]), 1))
        # self._ck2004_boosting_photon_grid = np.nan*np.ones((len(self.atm_axes['ck2004'][0]), len(self.atm_axes['ck2004'][1]), len(self.atm_axes['ck2004'][2]), len(self.atm_axes['ck2004'][3]), 1))

        if include_extinction:
            self.ext_axes[atm] = (np.unique(teffs), np.unique(loggs), np.unique(abuns), np.unique(ebvs), np.unique(rvs))

            teffs = np.repeat(teffs, len(ebvs)*len(rvs))
            loggs = np.repeat(loggs, len(ebvs)*len(rvs))
            abuns = np.repeat(abuns, len(ebvs)*len(rvs))

            self.ext_energy_grid[atm] = np.nan*np.ones((len(self.ext_axes[atm][0]), len(self.ext_axes[atm][1]), len(self.ext_axes[atm][2]), len(self.ext_axes[atm][3]), len(self.ext_axes[atm][4]), 1))
            self.ext_photon_grid[atm] = np.copy(self.ext_energy_grid[atm])

            flat_energy = ext_energy.flat
            flat_photon = ext_photon.flat

            for i in range(nmodels*len(ebvs)*len(rvs)):
                t = (teffs[i] == self.ext_axes[atm][0], loggs[i] == self.ext_axes[atm][1], abuns[i] == self.ext_axes[atm][2], ebv_list[i] == self.ext_axes[atm][3], rv_list[i] == self.ext_axes[atm][4], 0)
                self.ext_energy_grid[atm][t] = flat_energy[i]
                self.ext_photon_grid[atm][t] = flat_photon[i]

            if f'{atm}:ext' not in self.content:
                self.content.append(f'{atm}:ext')

        # Set the limb (mu=0) to 0; in log this formally means flux density=1W/m3, but compared to ~10 that is
        # the typical table[:,:,:,1,:] value, for all practical purposes that is still 0.
        self.atm_energy_grid[atm][:,:,:,0,:][~np.isnan(self.atm_energy_grid[atm][:,:,:,1,:])] = 0.0
        self.atm_photon_grid[atm][:,:,:,0,:][~np.isnan(self.atm_photon_grid[atm][:,:,:,1,:])] = 0.0
        # self._ck2004_boosting_energy_grid[:,:,:,0,:] = 0.0
        # self._ck2004_boosting_photon_grid[:,:,:,0,:] = 0.0

        for i, int_energy in enumerate(ints_energy):
            self.atm_energy_grid[atm][teffs[int(i/len(mus))] == self.atm_axes[atm][0], loggs[int(i/len(mus))] == self.atm_axes[atm][1], abuns[int(i/len(mus))] == self.atm_axes[atm][2], mus[i%len(mus)] == self.atm_axes[atm][3], 0] = int_energy
        for i, int_photon in enumerate(ints_photon):
            self.atm_photon_grid[atm][teffs[int(i/len(mus))] == self.atm_axes[atm][0], loggs[int(i/len(mus))] == self.atm_axes[atm][1], abuns[int(i/len(mus))] == self.atm_axes[atm][2], mus[i%len(mus)] == self.atm_axes[atm][3], 0] = int_photon
        # for i, Bavg in enumerate(boostingE):
        #     self._ck2004_boosting_energy_grid[Teff[i] == self.atm_axes['ck2004'][0], logg[i] == self.atm_axes['ck2004'][1], abun[i] == self.atm_axes['ck2004'][2], mu[i] == self.atm_axes['ck2004'][3], 0] = Bavg
        # for i, Bavg in enumerate(boostingP):
        #     self._ck2004_boosting_photon_grid[Teff[i] == self.atm_axes['ck2004'][0], logg[i] == self.atm_axes['ck2004'][1], abun[i] == self.atm_axes['ck2004'][2], mu[i] == self.atm_axes['ck2004'][3], 0] = Bavg

        # Impute if requested:
        if impute:
            if verbose:
                print('Imputing the grids...')
            for grid in (self.atm_energy_grid[atm], self.atm_photon_grid[atm]):
                for i in range(len(self.atm_axes[atm][-1])):
                    ndpolator.impute_grid(self.atm_axes[atm][:-1], grid[...,i,:])

        # Build the table of non-null indices for the nearest neighbor lookup:
        self.indices[atm] = np.argwhere(~np.isnan(self.atm_photon_grid[atm][...,-1,:]))
        non_nan_vertices = np.array([ [self.atm_axes[atm][i][self.indices[atm][k][i]] for i in range(len(self.atm_axes[atm])-1)] for k in range(len(self.indices[atm]))])
        self.nntree[atm] = cKDTree(non_nan_vertices, copy_data=True)

        # Set up the blending region:
        self.blending_region[atm] = brs
        self.mapper[atm] = lambda v: ndpolator.map_to_cube(v, self.atm_axes[atm][:-1], self.blending_region[atm])

        # Store all inferior corners for quick nearest neighbor lookup:
        raxes = self.atm_axes[atm][:-1]
        subgrid = self.atm_photon_grid[atm][...,-1,:]
        self.ics[atm] = np.array([(i, j, k) for i in range(0,len(raxes[0])-1) for j in range(0,len(raxes[1])-1) for k in range(0,len(raxes[2])-1) if ~np.any(np.isnan(subgrid[i:i+2,j:j+2,k:k+2]))])

        self.ndp[f'imu@photon@{atm}'] = ndpolator.Ndpolator(self.atm_axes[atm], self.atm_photon_grid[atm])
        self.ndp[f'imu@energy@{atm}'] = ndpolator.Ndpolator(self.atm_axes[atm], self.atm_energy_grid[atm])

        if f'{atm}:Imu' not in self.content:
            self.content.append(f'{atm}:Imu')

    def _ld(self, mu=1.0, ld_coeffs=np.array([[0.5]]), ld_func='linear'):
        ld_coeffs = np.atleast_2d(ld_coeffs)

        if ld_func == 'linear':
            return 1-ld_coeffs[:,0]*(1-mu)
        elif ld_func == 'logarithmic':
            return 1-ld_coeffs[:,0]*(1-mu)-ld_coeffs[:,1]*mu*np.log(np.maximum(mu, 1e-6))
        elif ld_func == 'square_root':
            return 1-ld_coeffs[:,0]*(1-mu)-ld_coeffs[:,1]*(1-np.sqrt(mu))
        elif ld_func == 'quadratic':
            return 1-ld_coeffs[:,0]*(1-mu)-ld_coeffs[:,1]*(1-mu)*(1-mu)
        elif ld_func == 'power':
            return 1-ld_coeffs[:,0]*(1-np.sqrt(mu))-ld_coeffs[:,1]*(1-mu)-ld_coeffs[:,2]*(1-mu*np.sqrt(mu))-ld_coeffs[:,3]*(1.0-mu*mu)
        else:
            raise NotImplementedError(f'ld_func={ld_func} is not supported.')

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

    def compute_ldcoeffs(self, ldatm, weighting='uniform', plot_diagnostics=False):
        """
        Computes limb darkening coefficients for linear, log, square root,
        quadratic and power laws.

        Arguments
        ----------
        * `ldatm` (string): model atmosphere for the limb darkening
          coefficients
        * `weighting` (string, optional, default='uniform'): determines how
            data points should be weighted.
            * 'uniform':  do not apply any per-point weighting
            * 'interval': apply weighting based on the interval widths
        * `plot_diagnostics` (bool, optional, default=False): should
          diagnostic graphs be plotted at compute time
        """

        if f'{ldatm}:Imu' not in self.content:
            raise RuntimeError(f'atm={ldatm} intensities are not found in the {self.pbset}:{self.pbname} passband.')

        self.ld_energy_grid[ldatm] = np.nan*np.ones((len(self.atm_axes[ldatm][0]), len(self.atm_axes[ldatm][1]), len(self.atm_axes[ldatm][2]), 11))
        self.ld_photon_grid[ldatm] = np.nan*np.ones((len(self.atm_axes[ldatm][0]), len(self.atm_axes[ldatm][1]), len(self.atm_axes[ldatm][2]), 11))
        mus = self.atm_axes[ldatm][3] # starts with 0
        if weighting == 'uniform':
            sigma = np.ones(len(mus))
        elif weighting == 'interval':
            delta = np.concatenate( (np.array((mus[1]-mus[0],)), mus[1:]-mus[:-1]) )
            sigma = 1./np.sqrt(delta)
        else:
            raise ValueError(f'weighting={weighting} is not supported.')

        for Tindex in range(len(self.atm_axes[ldatm][0])):
            for lindex in range(len(self.atm_axes[ldatm][1])):
                for mindex in range(len(self.atm_axes[ldatm][2])):
                    IsE = 10**self.atm_energy_grid[ldatm][Tindex,lindex,mindex,:].flatten()
                    fEmask = np.isfinite(IsE)
                    if len(IsE[fEmask]) <= 1:
                        continue
                    IsE /= IsE[fEmask][-1]

                    cElin,  pcov = cfit(f=self._ldlaw_lin,    xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5])
                    cElog,  pcov = cfit(f=self._ldlaw_log,    xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cEsqrt, pcov = cfit(f=self._ldlaw_sqrt,   xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cEquad, pcov = cfit(f=self._ldlaw_quad,   xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cEnlin, pcov = cfit(f=self._ldlaw_nonlin, xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5, 0.5, 0.5, 0.5])
                    self.ld_energy_grid[ldatm][Tindex, lindex, mindex] = np.hstack((cElin, cElog, cEsqrt, cEquad, cEnlin))

                    IsP = 10**self.atm_photon_grid[ldatm][Tindex,lindex,mindex,:].flatten()
                    fPmask = np.isfinite(IsP)
                    IsP /= IsP[fPmask][-1]

                    cPlin,  pcov = cfit(f=self._ldlaw_lin,    xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5])
                    cPlog,  pcov = cfit(f=self._ldlaw_log,    xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cPsqrt, pcov = cfit(f=self._ldlaw_sqrt,   xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cPquad, pcov = cfit(f=self._ldlaw_quad,   xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cPnlin, pcov = cfit(f=self._ldlaw_nonlin, xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5, 0.5, 0.5, 0.5])
                    self.ld_photon_grid[ldatm][Tindex, lindex, mindex] = np.hstack((cPlin, cPlog, cPsqrt, cPquad, cPnlin))

                    if plot_diagnostics:
                        # TODO: needs revision
                        if Tindex == 10 and lindex == 9 and mindex == 5:
                            print(self.atm_axes[ldatm][0][Tindex], self.atm_axes[ldatm][1][lindex], self.atm_axes[ldatm][2][mindex])
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

        if f'{ldatm}:ld' not in self.content:
            self.content.append(f'{ldatm}:ld')

    def export_phoenix_atmtab(self):
        """
        Exports PHOENIX intensity table to a PHOEBE legacy compatible format.
        """

        teffs = self.atm_axes['phoenix'][0]
        tlow, tup = teffs[0], teffs[-1]
        trel = (teffs-tlow)/(tup-tlow)

        for abun in range(len(self.atm_axes['phoenix'][2])):
            for logg in range(len(self.atm_axes['phoenix'][1])):
                logI = self.atm_energy_grid['phoenix'][:,logg,abun,-1,0]+1 # +1 to take care of WD units

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

    def export_legacy_ldcoeffs(self, models, atm='ck2004', filename=None, intens_weighting='photon'):
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
        * `intens_weighting`
        """

        axes = self.atm_axes[atm]
        grid = self.ld_photon_grid[atm] if intens_weighting == 'photon' else self.ld_energy_grid[atm]

        if filename is not None:
            import time
            f = open(filename, 'w')
            f.write('# PASS_SET  %s\n' % self.pbset)
            f.write('# PASSBAND  %s\n' % self.pbname)
            f.write('# VERSION   1.0\n\n')
            f.write('# Exported from PHOEBE-2 passband on %s\n' % (time.ctime()))
            f.write('# The coefficients are computed for the %s-weighted regime from %s atmospheres.\n\n' % (intens_weighting, atm))

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

    def compute_ldints(self, ldatm):
        """
        Computes integrated limb darkening profiles for the passed `ldatm`.

        These are used for intensity-to-flux transformations. The evaluated
        integral is:

        ldint = 2 \int_0^1 Imu mu dmu

        Arguments
        ----------
        * `ldatm` (string): model atmosphere for the limb darkening calculation.
        """

        if f'{ldatm}:Imu' not in self.content:
            raise RuntimeError(f'atm={ldatm} intensities are not found in the {self.pbset}:{self.pbname} passband.')

        ldaxes = self.atm_axes[ldatm]
        ldtable = self.atm_energy_grid[ldatm]
        pldtable = self.atm_photon_grid[ldatm]

        self.ldint_energy_grid[ldatm] = np.nan*np.ones((len(ldaxes[0]), len(ldaxes[1]), len(ldaxes[2]), 1))
        self.ldint_photon_grid[ldatm] = np.nan*np.ones((len(ldaxes[0]), len(ldaxes[1]), len(ldaxes[2]), 1))

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

                    self.ldint_energy_grid[ldatm][a,b,c] = 2*ldint
                    self.ldint_photon_grid[ldatm][a,b,c] = 2*pldint

        if f'{ldatm}:ldint' not in self.content:
            self.content.append(f'{ldatm}:ldint')

    def interpolate_ldcoeffs(self, teffs=5772., loggs=4.43, abuns=0.0, ldatm='ck2004', ld_func='power', intens_weighting='photon', ld_extrapolation_method='none'):
        """
        Interpolate the passband-stored table of LD model coefficients.

        Arguments
        ------------
        * `teffs` (float or array, default=5772): effective temperature
        * `loggs` (float or array, default=4.43): surface gravity in cgs
        * `abuns` (float or array, default=0.0): log-abundance in solar log-abundances
        * `ldatm` (string, default='ck2004'): limb darkening table: 'ck2004' or 'phoenix'
        * `ld_func` (string, default='power'): limb darkening fitting function: 'linear',
          'logarithmic', 'square_root', 'quadratic', 'power' or 'all'
        * `intens_weighting` (string, optional, default='photon'):
        * `ld_extrapolation_method` (string, optional, default='none'): extrapolation mode:
            'none', 'nearest', 'linear'

        Returns
        --------
        * (list or None) list of limb-darkening coefficients or None if 'ck2004:ld'
            is not available in <phoebe.atmospheres.passbands.Passband.content>
            (see also <phoebe.atmospheres.passbands.Passband.compute_ldcoeffs>)
            or if `ld_func` is not recognized.
        """

        s = {
            'linear': np.s_[:,:1],
            'logarithmic': np.s_[:,1:3],
            'square_root': np.s_[:,3:5],
            'quadratic': np.s_[:,5:7],
            'power': np.s_[:,7:11],
            'all': np.s_[:,:]
        }

        if ld_func not in s.keys():
            raise ValueError(f'ld_func={ld_func} is invalid; valid options are {s.keys()}.')

        if f'{ldatm}:ld' not in self.content:
            raise ValueError(f'Limb darkening coefficients for ldatm={ldatm} are not available; please compute them first.')

        axes = self.atm_axes[ldatm][:-1]
        table = self.ld_photon_grid[ldatm] if intens_weighting == 'photon' else self.ld_energy_grid[ldatm]

        req = ndpolator.tabulate((teffs, loggs, abuns))
        ndp = ndpolator.Ndpolator(axes, table)
        # ld_coeffs = libphoebe.interp(req, axes, table)
        ld_coeffs = ndp.interp(req, extrapolation_method=ld_extrapolation_method)

        return ld_coeffs[s[ld_func]]

    def interpolate_extinct(self, teffs=5772., loggs=4.43, abuns=0.0, atm='blackbody',  ebvs=0.0, rvs=3.1, intens_weighting='photon', extrapolation_method='none'):
        """
        Interpolates the passband-stored tables of extinction corrections

        Arguments
        ----------
        * `teffs` (float, optional, default=5772): effective temperature.
        * `loggs` (float, optional, default=4.43): log surface gravity
        * `abuns` (float, optional, default=0.0): abundance
        * `atm` (string, optional, default='blackbody'): atmosphere model.
        * `ebvs` (float, optional, default=0.0)
        * `rvs` (float, optional, default=3.1)
        * `intens_weighting`

        Returns
        ---------
        * extinction factor

        Raises
        --------
        * ValueError if `atm` is not supported.
        """

        if f'{atm}:ext' not in self.content:
            raise  ValueError(f"extinction factors for atm={atm} not found for the {self.pbset}:{self.pbname} passband.")

        axes = self.ext_axes[atm]
        if intens_weighting == 'photon':
            table = self.ext_photon_grid[atm]
        else:
            table = self.ext_energy_grid[atm]

        ndp = ndpolator.Ndpolator(axes, table)

        if atm == 'blackbody':
            req = ndp.tabulate((teffs, rvs, ebvs))
        else:
            req = ndp.tabulate((teffs, loggs, abuns, ebvs, rvs))

        extinct_factor = ndp.interp(req, extrapolation_method=extrapolation_method)
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

    def _log10_Inorm_extern_planckint(self, teffs):
        """
        Internal function to compute normal passband intensities using
        the external WD machinery that employs blackbody approximation.

        @teffs: effective temperature in K

        Returns: log10(Inorm)
        """

        log10_Inorm = libphoebe.wd_planckint(teffs, self.extern_wd_idx, self.wd_data["planck_table"])

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

    def _log10_Inorm(self, atm, teffs, loggs, abuns, intens_weighting='photon', atm_extrapolation_method='none', ld_extrapolation_method='none', blending_method='none', raise_on_nans=True, return_nanmask=False):
        """
        Computes normal emergent passband intensities for model atmospheres.

        Parameters
        ----------
        atm : string
            model atmosphere ('ck2004', 'phoenix', 'tmap')
        teffs : float or array
            effective temperature in K
        loggs : float or array
            surface gravity in cgs
        abuns : float or array
            heavy element abundance (metallicity), in log-solar
        intens_weighting : str, optional
            intensity weighting scheme, by default 'photon'
        atm_extrapolation_method : str, optional
            out-of-bounds intensity extrapolation method, by default 'none'
        ld_extrapolation_method : str, optional
            out-of-bounds limb darkening extrapolation method, by default
            'none'
        blending_method : str, optional
            out-of-bounds blending method, by default 'none'
        raise_on_nans : bool, optional
            should an error be raised on failed intensity lookup, by default
            True
        return_nanmask : bool, optional
            if an error is not raised, should a mask of non-value elements be
            returned, by default False

        Returns
        -------
        log10(intensity)
            interpolated (possibly extrapolated, blended) model atmosphre
            intensity

        Raises
        ------
        ValueError
            _description_
        """
        axes = self.atm_axes[atm][:-1]
        grid = self.atm_photon_grid[atm][...,-1,:] if intens_weighting == 'photon' else self.atm_energy_grid[atm][...,-1,:]
        ndp = ndpolator.Ndpolator(axes, grid)
        req = ndp.tabulate((teffs, loggs, abuns))
        log10_Inorm, nanmask = ndp.interp(req, raise_on_nans=raise_on_nans, return_nanmask=True, extrapolation_method=atm_extrapolation_method)
        # nanmask is a mask of elements that were nans before extrapolation.

        if ~np.any(nanmask):
            return (log10_Inorm, nanmask) if return_nanmask else log10_Inorm

        if blending_method == 'blackbody':
            log10_Inorm_bb = np.log10(self.Inorm(atm='blackbody', teffs=teffs[nanmask], loggs=loggs[nanmask], abuns=abuns[nanmask], ldatm=atm, ld_extrapolation_method=ld_extrapolation_method, intens_weighting=intens_weighting))
            nv, naxes = ndpolator.map_to_cube(req[nanmask], self.atm_axes[atm][:-1], self.blending_region[atm], return_naxes=True)

            log10_Inorm_bl = np.empty_like(teffs[nanmask])
            for si, selem in enumerate(nv):
                ic = [np.searchsorted(naxes[k], selem[k])-1 for k in range(len(naxes))]
                seps = (np.abs(self.ics[atm]-np.array(ic))).sum(axis=1)
                corners = np.argwhere(seps == seps.min()).flatten()

                blints_per_corner = []
                for corner in corners:
                    slc = tuple([slice(self.ics[atm][corner][i], self.ics[atm][corner][i]+2) for i in range(len(self.ics[atm][corner]))])
                    coords = [naxes[i][slc[i]] for i in range(len(naxes))]
                    verts = np.array(np.meshgrid(*coords)).T.reshape(-1, len(naxes))  # faster than itertools.product(*coords)
                    distance_vectors = selem-verts
                    distances = np.linalg.norm(distance_vectors, axis=1)
                    distance_vector = distance_vectors[distances.argmin()]

                    shift = ic-self.ics[atm][corner]
                    shift = shift != 0

                    if shift.sum() == 0:
                        raise ValueError('how did we get here?')

                    # project the vertex distance to the nearest hyperface/hyperedge:
                    distance_vector *= shift
                    distance = np.linalg.norm(distance_vector)

                    if distance > 1:
                        blints_per_corner.append(log10_Inorm_bb[si])
                        continue

                    alpha = blending_factor(distance)

                    blints_per_corner.append((1-alpha)*log10_Inorm_bb[si] + alpha*log10_Inorm[nanmask][si])
                    # print(f'distance={distance}, alpha={alpha}, bb={ints_bb[si]}, ck={ints_ck[nanmask][si]}, bl={blints_per_corner[-1]}')

                log10_Inorm_bl[si] = np.mean(blints_per_corner)

            log10_Inorm[nanmask] = log10_Inorm_bl[:,None]

        return (log10_Inorm, nanmask) if return_nanmask else log10_Inorm

    def Inorm(self, teffs=5772., loggs=4.43, abuns=0.0, atm='ck2004', ldatm='ck2004', ldint=None, ld_func='interp', ld_coeffs=None, intens_weighting='photon', atm_extrapolation_method='none', ld_extrapolation_method='none', blending_method='none', return_nanmask=False):
        """
        Computes normal emergent passband intensity.

        Possible atm/ldatm/ld_func/ld_coeffs combinations:

        | atm       | ldatm         | ld_func                 | ld_coeffs | intens_weighting | action                                                      |
        ------------|---------------|-------------------------|-----------|------------------|-------------------------------------------------------------|
        | blackbody | none          | *                       | none      | *                | raise error                                                 |
        | blackbody | none          | lin,log,quad,sqrt,power | *         | *                | use manual LD model                                         |
        | blackbody | supported atm | interp                  | none      | *                | interpolate from ck2004:Imu                                 |
        | blackbody | supported atm | interp                  | *         | *                | interpolate from ck2004:Imu but warn about unused ld_coeffs |
        | blackbody | supported atm | lin,log,quad,sqrt,power | none      | *                | interpolate ld_coeffs from ck2004:ld                        |
        | blackbody | supported atm | lin,log,quad,sqrt,power | *         | *                | use manual LD model but warn about unused ldatm             |
        | planckint | *             | *                       | *         | photon           | raise error                                                 |
        | atmx      | *             | *                       | *         | photon           | raise error                                                 |
        | ck2004    |               |                         |           |                  |                                                             |
        | phoenix   |               |                         |           |                  |                                                             |
        | tmap      |               |                         |           |                  |                                                             |

        Arguments
        ----------
        * `teffs` (float or array, optional, default=5772): effective
          temperature of the emitting surface element(s), in Kelvin
        * `loggs` (float or array, optional, default=4.43): surface gravity of
          the emitting surface element(s), in log(cm/s^2)
        * `abuns` (float or array, optional, default=0.0): chemical
          abundance/metallicity of the emitting surface element(s), in solar
          dex
        * `atm` (string, optional, default='ck2004'): model atmosphere to be
          used for calculation
        * `ldatm` (string, optional, default='ck2004'): model atmosphere to be
          used for limb darkening coefficients
        * `ldint` (string, optional, default=None): integral of the limb
            darkening function, \int_0^1 \mu L(\mu) d\mu. Its general role is
            to convert intensity to flux. In this method, however, it is only
            needed for blackbody atmospheres because they are not
            limb-darkened (i.e. the blackbody intensity is the same
            irrespective of \mu), so we need to *divide* by ldint to ascertain
            the correspondence between luminosity, effective temperature and
            fluxes once limb darkening correction is applied at flux
            integration time. If None, and if `atm=='blackbody'`, it will be
            computed from `ld_func` and `ld_coeffs`.
        * `ld_func` (string, optional, default='interp') limb darkening
            function.  One of: linear, sqrt, log, quadratic, power, interp.
        * `ld_coeffs` (list, optional, default=None): limb darkening
            coefficients for the corresponding limb darkening function,
            `ld_func`. If None, the coefficients are interpolated from the
            corresponding table. List length needs to correspond to the
            `ld_func`: 1 for linear, 2 for sqrt, log and quadratic, and 4 for
            power.
        * `intens_weighting` (string, optional, default='photon'): photon/energy
          switch
        * `atm_extraplation_method` (string, optional, default='none'): the
          method of intensity extrapolation and off-the-grid blending with
          blackbody atmosheres ('none', 'nearest', 'linear')
        * `ld_extrapolation_method` (string, optional, default='none'): the
          method of limb darkening extrapolation ('none', 'nearest' or
          'linear')
        * `blending_method` (string, optional, default='none'): whether to
          blend model atmosphere with blackbody ('none' or 'blackbody')
        * `return_nanmask` (boolean, optional, default=False): should a mask
          of off-grid intensities be returned in addition to intensities.

        Returns
        ----------
        * (array) normal emargent passband intensities, or:
        * (tuple) normal emargent passband intensities and a nan mask.

        Raises
        ----------
        * ValueError: if atmosphere parameters are out of bounds for the
          table.
        * NotImplementedError: if `ld_func` is not supported.
        """
        # import inspect
        # print(f'{teffs=}\n{loggs=}\n{abuns=}\n{atm=}\n{ldatm=}\n{ldint=}\n{ld_func=}\n{ld_coeffs=}\n{intens_weighting=}\n{atm_extrapolation_method=}\n{ld_extrapolation_method=}\n{blending_method=}\n{return_nanmask=}')
        # for fi in inspect.stack():
        #     print(fi)

        if atm not in ['blackbody', 'extern_planckint', 'extern_atmx', 'ck2004', 'phoenix', 'tmap']:
            raise RuntimeError(f'atm={atm} is not supported.')

        if ldatm not in ['none', 'ck2004', 'phoenix', 'tmap']:
            raise ValueError(f'ldatm={ldatm} is not supported.')

        if intens_weighting not in ['energy', 'photon']:
            raise ValueError(f'intens_weighting={intens_weighting} is not supported.')

        if blending_method not in ['none', 'blackbody']:
            raise ValueError(f'blending_method={blending_method} is not supported.')

        raise_on_nans = True if atm_extrapolation_method == 'none' else False

        # tabulate requested atmosphere parameters:
        req = ndpolator.tabulate((teffs, loggs, abuns))

        if atm == 'blackbody' and 'blackbody:Inorm' in self.content:
            # check if the required tables for the chosen ldatm are available:
            if ldatm == 'none' and ld_coeffs is None:
                raise ValueError("ld_coeffs must be passed when ldatm='none'.")
            if ld_func == 'interp' and f'{ldatm}:Imu' not in self.content:
                raise RuntimeError(f'passband {self.pbset}:{self.pbname} does not contain specific intensities for ldatm={ldatm}.')
            if ld_func != 'interp' and ld_coeffs is None and f'{ldatm}:ld' not in self.content:
                raise RuntimeError(f'passband {self.pbset}:{self.pbname} does not contain limb darkening coefficients for ldatm={ldatm}.')
            if blending_method == 'blackbody':
                raise ValueError(f'the combination of atm={atm} and blending_method={blending_method} is not valid.')

            # calculate intensities:
            if intens_weighting == 'photon':
                intensities = 10**self._log10_Inorm_bb_photon(req[:,0]).reshape(-1, 1)
            else:  # if intens_weighting=='energy':
                intensities = 10**self._log10_Inorm_bb_energy(req[:,0]).reshape(-1, 1)

            if ld_func != 'interp' and ld_coeffs is None:
                ld_coeffs = self.interpolate_ldcoeffs(teffs, loggs, abuns, ldatm, ld_func=ld_func, intens_weighting=intens_weighting, ld_extrapolation_method=ld_extrapolation_method)

            if ldint is None:
                ldint = self.ldint(teffs=teffs, loggs=loggs, abuns=abuns, ldatm=ldatm, ld_func=ld_func, ld_coeffs=ld_coeffs, intens_weighting=intens_weighting, ld_extrapolation_method=ld_extrapolation_method, raise_on_nans=raise_on_nans)

            intensities /= ldint.reshape(-1, 1)

        elif atm == 'extern_planckint' and 'extern_planckint:Inorm' in self.content:
            if intens_weighting == 'photon':
                raise ValueError(f'the combination of atm={atm} and intens_weighting={intens_weighting} is not supported.')
            # TODO: add all other exceptions

            intensities = 10**(self._log10_Inorm_extern_planckint(req[:,0])-1)  # -1 is for cgs -> SI
            if ldint is None:
                ldint = self.ldint(teffs=teffs, loggs=loggs, abuns=abuns, ldatm=ldatm, ld_func=ld_func, ld_coeffs=ld_coeffs, intens_weighting=intens_weighting, ld_extrapolation_method=ld_extrapolation_method, raise_on_nans=raise_on_nans)
            intensities /= ldint

        elif atm == 'extern_atmx' and 'extern_atmx:Inorm' in self.content:
            if intens_weighting == 'photon':
                raise ValueError(f'the combination of atm={atm} and intens_weighting={intens_weighting} is not supported.')
            # TODO: add all other exceptions

            intensities = 10**(self._log10_Inorm_extern_atmx(req[:,0], req[:,1], req[:,2])-1)  # -1 is for cgs -> SI

        else:
            if f'{atm}:Imu' not in self.content:
                raise ValueError(f'atm={atm} tables are not available in the {self.pbset}:{self.pbname} passband.')

            if return_nanmask:
                intensities, nanmask = self._log10_Inorm(atm=atm, teffs=teffs, loggs=loggs, abuns=abuns, intens_weighting=intens_weighting, atm_extrapolation_method=atm_extrapolation_method, ld_extrapolation_method=ld_extrapolation_method, blending_method=blending_method, raise_on_nans=raise_on_nans, return_nanmask=return_nanmask)
                intensities = (10**intensities, nanmask)
            else:
                intensities = 10**self._log10_Inorm(atm=atm, teffs=teffs, loggs=loggs, abuns=abuns, intens_weighting=intens_weighting, atm_extrapolation_method=atm_extrapolation_method, ld_extrapolation_method=ld_extrapolation_method, blending_method=blending_method, raise_on_nans=raise_on_nans, return_nanmask=return_nanmask)

        # print(f'{intensities.flatten()=}')
        return intensities

    def _log10_Imu(self, atm, teffs, loggs, abuns, mus, intens_weighting='photon', atm_extrapolation_method='none', ld_extrapolation_method='none', blending_method='none', raise_on_nans=True, return_nanmask=False):
        """
        Computes specific emergent passband intensities for model atmospheres.

        Parameters
        ----------
        atm : string
            model atmosphere ('ck2004', 'phoenix', 'tmap')
        teffs : float or array
            effective temperature in K
        loggs : float or array
            surface gravity in cgs
        abuns : float or array
            heavy element abundance (metallicity), in log-solar
        mus : float or array
            specific angles mu=cos(theta), where 0 is limb and 1 is disk
            center
        intens_weighting : str, optional
            intensity weighting scheme, by default 'photon'
        atm_extrapolation_method : str, optional
            out-of-bounds intensity extrapolation method, by default 'none'
        ld_extrapolation_method : str, optional
            out-of-bounds limb darkening extrapolation method, by default
            'none'
        blending_method : str, optional
            out-of-bounds blending method, by default 'none'
        raise_on_nans : bool, optional
            should an error be raised on failed intensity lookup, by default
            True
        return_nanmask : bool, optional
            if an error is not raised, should a mask of non-value elements be
            returned, by default False

        Returns
        -------
        log10(intensity)
            interpolated (possibly extrapolated, blended) model atmosphre
            intensity

        Raises
        ------
        ValueError
            _description_
        """

        # a temporary shortcut for testing purposes only:
        if atm == 'ck2004':
            ndp = self.ndp[f'imu@{intens_weighting}@{atm}']
        else:
            axes = self.atm_axes[atm]
            grid = self.atm_photon_grid[atm] if intens_weighting == 'photon' else self.atm_energy_grid[atm]
            ndp = ndpolator.Ndpolator(axes, grid)

        req = ndp.tabulate((teffs, loggs, abuns, mus))
        log10_Imu, nanmask = ndp.interp(req, raise_on_nans=raise_on_nans, return_nanmask=True, extrapolation_method=atm_extrapolation_method)

        if ~np.any(nanmask):
            return (log10_Imu, nanmask) if return_nanmask else log10_Imu

        if blending_method == 'blackbody':
            log10_Imu_bb = np.log10(self.Imu(atm='blackbody', teffs=teffs[nanmask], loggs=loggs[nanmask], abuns=abuns[nanmask], mus=mus[nanmask], ldatm=atm, ld_extrapolation_method=ld_extrapolation_method, intens_weighting=intens_weighting))
            nv, naxes = ndpolator.map_to_cube(req[nanmask], axes, self.blending_region[atm], return_naxes=True)

            log10_Imu_bl = np.empty_like(teffs[nanmask])
            for si, selem in enumerate(nv):
                ic = [np.searchsorted(naxes[k], selem[k])-1 for k in range(len(naxes))]
                seps = (np.abs(self.ics[atm]-np.array(ic))).sum(axis=1)
                corners = np.argwhere(seps == seps.min()).flatten()

                blints_per_corner = []
                for corner in corners:
                    slc = tuple([slice(self.ics[atm][corner][i], self.ics[atm][corner][i]+2) for i in range(len(self.ics[atm][corner]))])
                    coords = [naxes[i][slc[i]] for i in range(len(naxes))]
                    verts = np.array(np.meshgrid(*coords)).T.reshape(-1, len(naxes))  # faster than itertools.product(*coords)
                    distance_vectors = selem-verts
                    distances = np.linalg.norm(distance_vectors, axis=1)
                    distance_vector = distance_vectors[distances.argmin()]

                    shift = ic-self.ics[atm][corner]
                    shift = shift != 0

                    if shift.sum() == 0:
                        raise ValueError('how did we get here?')

                    # project the vertex distance to the nearest hyperface/hyperedge:
                    distance_vector *= shift
                    distance = np.linalg.norm(distance_vector)

                    if distance > 1:
                        blints_per_corner.append(log10_Imu_bb[si])
                        continue

                    alpha = blending_factor(distance)

                    blints_per_corner.append((1-alpha)*log10_Imu_bb[si] + alpha*log10_Imu[nanmask][si])
                    # print(f'distance={distance}, alpha={alpha}, bb={ints_bb[si]}, ck={ints_ck[nanmask][si]}, bl={blints_per_corner[-1]}')

                log10_Imu_bl[si] = np.mean(blints_per_corner)

            log10_Imu[nanmask] = log10_Imu_bl[:,None]

        if ~np.any(nanmask):
            return (log10_Imu, nanmask) if return_nanmask else log10_Imu

    def Imu(self, teffs=5772., loggs=4.43, abuns=0.0, mus=1.0, atm='ck2004', ldatm='ck2004', ldint=None, ld_func='interp', ld_coeffs=None, intens_weighting='photon', atm_extrapolation_method='none', ld_extrapolation_method='none', blending_method='none', return_nanmask=False):
        """
        Computes specific emergent passband intensities.

        Arguments
        ----------
        * `teffs` (float or array, optional, default=5772): effective
          temperature of the emitting surface element(s), in Kelvin
        * `loggs` (float or array, optional, default=4.43): surface gravity of
          the emitting surface element(s), in log(cm/s^2)
        * `abuns` (float or array, optional, default=0.0): chemical
          abundance/metallicity of the emitting surface element(s), in solar
          dex
        * `mus` (float or array, optional, default=1.0): specific angles
          mu=cos(theta), where mu=0 is the limb and mu=1 is the disk center.
        * `atm` (string, optional, default='ck2004'): model atmosphere to be
          used for calculation
        * `ldatm` (string, optional, default='ck2004'): model atmosphere to be
          used for limb darkening coefficients
        * `ldint` (string, optional, default=None): integral of the limb
            darkening function, \int_0^1 \mu L(\mu) d\mu. Its general role is
            to convert intensity to flux. In this method, however, it is only
            needed for blackbody atmospheres because they are not
            limb-darkened (i.e. the blackbody intensity is the same
            irrespective of \mu), so we need to *divide* by ldint to ascertain
            the correspondence between luminosity, effective temperature and
            fluxes once limb darkening correction is applied at flux
            integration time. If None, and if `atm=='blackbody'`, it will be
            computed from `ld_func` and `ld_coeffs`.
        * `ld_func` (string, optional, default='interp') limb darkening
            function.  One of: linear, sqrt, log, quadratic, power, interp.
        * `ld_coeffs` (list, optional, default=None): limb darkening
            coefficients for the corresponding limb darkening function,
            `ld_func`. If None, the coefficients are interpolated from the
            corresponding table. List length needs to correspond to the
            `ld_func`: 1 for linear, 2 for sqrt, log and quadratic, and 4 for
            power.
        * `intens_weighting` (string, optional, default='photon'): photon/energy
          switch
        * `atm_extraplation_method` (string, optional, default='none'): the
          method of intensity extrapolation and off-the-grid blending with
          blackbody atmosheres ('none', 'nearest', 'linear')
        * `ld_extrapolation_method` (string, optional, default='none'): the
          method of limb darkening extrapolation ('none', 'nearest' or
          'linear')
        * `blending_method` (string, optional, default='none'): whether to
          blend model atmosphere with blackbody ('none' or 'blackbody')
        * `return_nanmask` (boolean, optional, default=False): should a mask
          of off-grid intensities be returned in addition to intensities.

        Returns
        ----------
        * (array) specific emargent passband intensities, or:
        * (tuple) specific emargent passband intensities and a nan mask.

        Raises
        ----------
        * ValueError: if atmosphere parameters are out of bounds for the
          table.
        * NotImplementedError: if `ld_func` is not supported.
        """

        if atm not in ['blackbody', 'extern_planckint', 'extern_atmx', 'ck2004', 'phoenix', 'tmap']:
            raise RuntimeError(f'atm={atm} is not supported.')

        if ldatm not in ['none', 'ck2004', 'phoenix', 'tmap']:
            raise ValueError(f'ldatm={ldatm} is not supported.')

        raise_on_nans = True if atm_extrapolation_method == 'none' else False

        if ld_func == 'interp':
            # 'interp' works only for model atmospheres:
            if atm not in ['ck2004', 'phoenix', 'tmap']:
                raise ValueError(f"atm={atm} cannot be used with ld_func={ld_func}.")

            if atm not in self.content and f'{atm}:Imu' not in self.content:
                raise ValueError(f'atm={atm} tables are not available in the {self.pbset}:{self.pbname} passband.')

            if return_nanmask:
                intensities, nanmask = self._log10_Imu(atm=atm, teffs=teffs, loggs=loggs, abuns=abuns, mus=mus, intens_weighting=intens_weighting, atm_extrapolation_method=atm_extrapolation_method, ld_extrapolation_method=ld_extrapolation_method, blending_method=blending_method, raise_on_nans=raise_on_nans, return_nanmask=return_nanmask)
                intensities = (10**intensities, nanmask)
            else:
                intensities = 10**self._log10_Imu(atm=atm, teffs=teffs, loggs=loggs, abuns=abuns, mus=mus, intens_weighting=intens_weighting, atm_extrapolation_method=atm_extrapolation_method, ld_extrapolation_method=ld_extrapolation_method, blending_method=blending_method, raise_on_nans=raise_on_nans, return_nanmask=return_nanmask)

            return intensities

        if ld_coeffs is None:
            # LD function can be passed without coefficients; in that
            # case we need to interpolate them from the tables.
            ld_coeffs = self.interpolate_ldcoeffs(teffs, loggs, abuns, ldatm, ld_func, intens_weighting=intens_weighting, ld_extrapolation_method=ld_extrapolation_method)

        Inorm = self.Inorm(
            teffs=teffs,
            loggs=loggs,
            abuns=abuns,
            atm=atm,
            ldatm=ldatm,
            ldint=ldint,
            ld_func=ld_func,
            ld_coeffs=ld_coeffs,
            intens_weighting=intens_weighting,
            atm_extrapolation_method=atm_extrapolation_method,
            ld_extrapolation_method=ld_extrapolation_method,
            blending_method=blending_method
        ).flatten()

        ld = self._ld(ld_func=ld_func, mu=mus, ld_coeffs=ld_coeffs)
        retval = Inorm * ld

        return retval

    def ldint(self, teffs=None, loggs=None, abuns=None, ldatm=None, ld_func='linear', ld_coeffs=np.array([[0.5]]), intens_weighting='photon', ld_extrapolation_method='none', raise_on_nans=True):
        """
        Computes ldint value for the given `ld_func` and `ld_coeffs`.

        Arguments
        ----------
        * `teffs` (float or array, required): effective temperatures for ldint computation
        * `loggs` (float or array, required): surface gravities for ldint computation
        * `abuns` (float or array, required): chemical abundances for ldint computation
        * `ldatm` (string, optional, required with ld_func='interp'): model atmosphere to
            be used for interpolation
        * `ld_func` (string, optional, default='linear'): limb darkening function
        * `ld_coeffs` (array, optional, default=[[0.5]]): limb darkening coefficients
        * `intens_weighting` (string, optional, default='photon'): intensity weighting
            scheme
        * `ld_extrapolation_method` (string, optional, default='none'): off-grid
            extrapolation method when ld_func='interp'; one of ['none', 'nearest',
            'linear']
        * `raise_on_nans` (boolean, optional, default=True): in the off-grid case when
            ld_func='interp', should an error be raised

        Returns
        -------
        * (array) ldint values

        Specific behavior:

        if ldatm is one of ['none', 'blackbody', 'extern_planckint', 'extern_atmx']:
            ld_func == 'interp': raise an error
            ld_func != 'interp' and ld_coeffs == 'none': raise an error
            ld_func != 'interp' and ld_coeffs is a (1, M) array: broadcast to (N, M) array
            ld_func != 'interp' and ld_coeffs is a (N, M) array: do nothing
        if ldatm is one of ['ck2004', 'phoenix', 'tmap']:
            ld_func == 'interp': do nothing
            ld_func != 'interp' and ld_coeffs == 'none': interpolate ld_coeffs
            ld_func != 'interp' and ld_coeffs is a (1, M) array: broadcast to (N, M) array
            ld_func != 'interp' and ld_coeffs is a (N, M) array: do nothing
        """

        # print(f'{teffs=}\n{loggs=}\n{abuns=}\n{ldatm=}\n{ld_func=}\n{ld_coeffs=}\n{intens_weighting=}\n{ld_extrapolation_method=}\n{raise_on_nans=}')

        if ldatm in [None, 'none', 'blackbody', 'extern_planckint', 'extern_atmx'] and ld_func == 'interp':
            raise ValueError(f'ldatm={ldatm} cannot be used with ld_func={ld_func}.')

        if ldatm in [None, 'none', 'blackbody', 'extern_planckint', 'extern_atmx'] and ld_func != 'interp' and ld_coeffs is None:
            raise ValueError(f'ldatm={ldatm} with ld_func={ld_func} cannot be used without passing ld_coeffs.')

        if ld_func != 'interp' and ld_coeffs is None:
            ld_coeffs = self.interpolate_ldcoeffs(teffs, loggs, abuns, ldatm, ld_func, intens_weighting=intens_weighting, ld_extrapolation_method=ld_extrapolation_method)

        req = ndpolator.tabulate((teffs, loggs, abuns))

        if ld_coeffs is not None:
            ld_coeffs = np.atleast_2d(ld_coeffs)
            ld_coeffs = np.broadcast_to(ld_coeffs, (len(req), ld_coeffs.shape[1]))  # (1, M) -> (N, M) or (N, M) -> (N, M)

        if ld_func == 'linear':
            ldints = 1-ld_coeffs[:,0]/3
        elif ld_func == 'logarithmic':
            ldints = 1-ld_coeffs[:,0]/3+2.*ld_coeffs[:,1]/9
        elif ld_func == 'square_root':
            ldints = 1-ld_coeffs[:,0]/3-ld_coeffs[:,1]/5
        elif ld_func == 'quadratic':
            ldints = 1-ld_coeffs[:,0]/3-ld_coeffs[:,1]/6
        elif ld_func == 'power':
            ldints = 1-ld_coeffs[:,0]/5-ld_coeffs[:,1]/3-3.*ld_coeffs[:,2]/7-ld_coeffs[:,3]/2
        elif ld_func == 'interp':
            axes = self.atm_axes[ldatm][:-1]
            grid = self.ldint_photon_grid[ldatm] if intens_weighting == 'photon' else self.ldint_energy_grid[ldatm]
            ndp = ndpolator.Ndpolator(axes, grid)

            ldints = ndp.interp(req, extrapolation_method=ld_extrapolation_method, return_nanmask=False, raise_on_nans=raise_on_nans)
        else:
            raise NotImplementedError(f'ld_func={ld_func} is not supported')

        return ldints

    def _bindex_blackbody(self, Teff, intens_weighting='photon'):
        """
        Computes the mean boosting index using blackbody atmosphere:

        B_pb^E = \int_\lambda I(\lambda) P(\lambda) B(\lambda) d\lambda / \int_\lambda I(\lambda) P(\lambda) d\lambda
        B_pb^P = \int_\lambda \lambda I(\lambda) P(\lambda) B(\lambda) d\lambda / \int_\lambda \lambda I(\lambda) P(\lambda) d\lambda

        Superscripts E and P stand for energy and photon, respectively.

        Arguments
        ----------
        * `Teff` (float/array): effective temperature in K
        * `intens_weighting`

        Returns
        ------------
        * mean boosting index using blackbody atmosphere.
        """

        if intens_weighting == 'photon':
            num   = lambda w: w*self._planck(w, Teff)*self.ptf(w)*self._planck_spi(w, Teff)
            denom = lambda w: w*self._planck(w, Teff)*self.ptf(w)
            return integrate.quad(num, self.wl[0], self.wl[-1], epsabs=1e10, epsrel=1e-8)[0]/integrate.quad(denom, self.wl[0], self.wl[-1], epsabs=1e10, epsrel=1e-6)[0]
        else:
            num   = lambda w: self._planck(w, Teff)*self.ptf(w)*self._planck_spi(w, Teff)
            denom = lambda w: self._planck(w, Teff)*self.ptf(w)
            return integrate.quad(num, self.wl[0], self.wl[-1], epsabs=1e10, epsrel=1e-8)[0]/integrate.quad(denom, self.wl[0], self.wl[-1], epsabs=1e10, epsrel=1e-6)[0]

    def _bindex_ck2004(self, req, atm, intens_weighting='photon'):
        grid = self._ck2004_boosting_photon_grid if intens_weighting == 'photon' else self._ck2004_boosting_energy_grid
        bindex = libphoebe.interp(req, self.atm_axes['ck2004'], grid).T[0]
        return bindex

    def bindex(self, teffs=5772., loggs=4.43, abuns=0.0, mus=1.0, atm='ck2004', intens_weighting='photon'):
        """
        """
        # TODO: implement phoenix boosting.
        raise NotImplementedError('Doppler boosting is currently offline for review.')

        req = ndpolator.tabulate((Teff, logg, abun, mu))

        if atm == 'ck2004':
            retval = self._bindex_ck2004(req, atm, intens_weighting)
        elif atm == 'blackbody':
            retval = self._bindex_blackbody(req[:,0], intens_weighting=intens_weighting)
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
        raise RuntimeError(f'failed to load passband at {fullpath}')
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
        raise ValueError("'{}' passband not available online".format(passband))

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

def Inorm_bol_bb(Teff=5772., logg=4.43, abun=0.0, atm='blackbody', intens_weighting='photon'):
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
    * `intens_weighting`

    Returns
    ---------
    * (float/array) float or array (depending on input types) of normal
        bolometric blackbody intensities.

    Raises
    --------
    * ValueError: if `atm` is anything other than `'blackbody'`.
    """

    if atm != 'blackbody':
        raise ValueError('atmosphere must be set to blackbody for Inorm_bol_bb.')

    if intens_weighting == 'photon':
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
        ptf='tables/ptf/bolometric.ptf',
        pbset='Bolometric',
        pbname='900-40000',
        effwl=1.955e-6,
        wlunits=u.m,
        calibrated=True,
        reference='Flat response to simulate bolometric throughput',
        version=2.0,
        comments=''
    )

    pb.compute_blackbody_intensities(include_extinction=False)

    pb.compute_intensities(atm='ck2004', path='tables/ck2004', impute=True, verbose=True)
    pb.compute_ldcoeffs(ldatm='ck2004')
    pb.compute_ldints(ldatm='ck2004')

    pb.compute_intensities(atm='phoenix', path='tables/phoenix', impute=True, verbose=True)
    pb.compute_ldcoeffs(ldatm='phoenix')
    pb.compute_ldints(ldatm='phoenix')

    pb.compute_intensities(atm='tmap', path='tables/tmap', impute=True, verbose=True)
    pb.compute_ldcoeffs(ldatm='tmap')
    pb.compute_ldints(ldatm='tmap')

    pb.save('bolometric.fits')

    pb = Passband(
        ptf='tables/ptf/johnson_v.ptf',
        pbset='Johnson',
        pbname='V',
        effwl=5500.,
        wlunits=u.AA,
        calibrated=True,
        reference='Maiz Apellaniz (2006), AJ 131, 1184',
        version=2.0,
        comments=''
    )

    pb.compute_blackbody_intensities(include_extinction=True)
#    pb.compute_bb_reddening(verbose=True)

    pb.compute_intensities(atm='ck2004', path='tables/ck2004', impute=True, include_extinction=True, verbose=True)
    pb.compute_ldcoeffs(ldatm='ck2004')
    pb.compute_ldints(ldatm='ck2004')
#    pb.compute_ck2004_reddening(path='tables/ck2004', verbose=True)

    pb.compute_intensities(atm='phoenix', path='tables/phoenix', impute=True, include_extinction=True, verbose=True)
    pb.compute_ldcoeffs(ldatm='phoenix')
    pb.compute_ldints(ldatm='phoenix')
#    pb.compute_phoenix_reddening(path='tables/phoenix', verbose=True)

    pb.compute_intensities(atm='tmap', path='tables/tmap', include_extinction=True, impute=True, verbose=True)
    pb.compute_ldcoeffs(ldatm='tmap')
    pb.compute_ldints(ldatm='tmap')
#    pb.compute_tmap_reddening(path='tables/tmap', verbose=True)

    pb.import_wd_atmcof('tables/wd/atmcofplanck.dat', 'tables/wd/atmcof.dat', 7)

    pb.save('johnson_v.fits')
