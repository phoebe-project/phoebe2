from phoebe import __version__ as phoebe_version
from phoebe import conf, mpi
from phoebe.utils import _bytes
from tqdm import tqdm
from itertools import product

import ndpolator

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
atm_tables = {
    'ck2004': 'CK',
    'phoenix': 'PH',
    'tmap_sdO': 'TS',
    'tmap_DA': 'TA',
    'tmap_DAO': 'TM',
    'tmap_DO': 'TO'
}

supported_atms = list(atm_tables.keys()) + ['blackbody', 'extern_atmx', 'extern_planckint']

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
    def __init__(self, ptf=None, pbset='Johnson', pbname='V',
                 wlunits=u.AA, calibrated=False, reference='', version=1.0,
                 comment=None, oversampling=1, ptf_order=3, from_file=False):
        """
        <phoebe.atmospheres.passbands.Passband> class holds data and tools for
        passband-related computations, such as blackbody intensity, model
        atmosphere intensity, etc.

        Step #1: initialize passband object

        ```py pb = Passband(ptf='JOHNSON.V', pbset='Johnson', pbname='V',
        wlunits=u.AA, calibrated=True, reference='ADPS', version=1.0) ```

        Step #2: compute intensities for blackbody radiation:

        ```py pb.compute_blackbody_intensities() ```

        Step #3: compute Castelli & Kurucz (2004) intensities. To do this, the
        tables/ck2004 directory needs to be populated with non-filtered
        intensities available for download from %static%/ck2004.tar.

        ```py atmdir =
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
        'tables/ck2004')) pb.compute_ck2004_response(atmdir) ```

        Step #4: -- optional -- import WD tables for comparison. This can only
        be done if the passband is in the list of supported passbands in WD.
        The WD index of the passband is passed to the import_wd_atmcof()
        function below as the last argument.

        ```py from phoebe.atmospheres import atmcof atmdir =
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
        'tables/wd')) atmcof.init(atmdir+'/atmcofplanck.dat',
        atmdir+'/atmcof.dat') pb.import_wd_atmcof(atmdir+'/atmcofplanck.dat',
        atmdir+'/atmcof.dat', 7) ```

        Step #5: save the passband file:

        ```py atmdir =
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
        'tables/passbands')) pb.save(atmdir + '/johnson_v.ptf') ```

        From now on you can use `pbset`:`pbname` as a passband qualifier, i.e.
        Johnson:V for the example above. Further details on supported model
        atmospheres are available by issuing:

        ```py pb.content ```

        see <phoebe.atmospheres.passbands.content>

        Arguments
        ----------
        * `ptf` (string or numpy array, optional, default=None): passband
          transmission; if str, assume it is a filename: a 2-column file with
          wavelength in `wlunits` and transmission in arbitrary units; if
          numpy array, it is a (N, 2)-shaped array that contains the same two
          columns.
        * `pbset` (string, optional, default='Johnson'): name of the passband
            set (i.e. Johnson).
        * `pbname` (string, optional, default='V'): name of the passband name
            (i.e. V).
        * `wlunits` (unit, optional, default=u.AA): wavelength units from
            astropy.units used in `ptf`.
        * `calibrated` (bool, optional, default=False): True if transmission
          is
            in true fractional light, False if it is in relative proportions.
        * `reference` (string, optional, default=''): passband transmission
          data
            reference (i.e. ADPS).
        * `version` (float, optional, default=1.0): file version.
        * `comment` (string or None, optional, default=None): any additional comment
            about the passband.
        * `oversampling` (int, optional, default=1): the multiplicative factor
            of PTF dispersion to attain higher integration accuracy.
        * `ptf_order` (int, optional, default=3): spline order for fitting
            the passband transmission function.
        * `from_file` (bool, optional, default=False): a switch that instructs
            the class instance to skip all calculations and load all data from
            the file passed to the
            <phoebe.atmospheres.passbands.Passband.load> method.

        Returns
        ---------
        * an instatiated <phoebe.atmospheres.passbands.Passband> object.
        """

        if "'" in pbset or '"' in pbset:
            raise ValueError("pbset cannot contain quotation marks")
        if "'" in pbname or '"' in pbname:
            raise ValueError("pbname cannot contain quotation marks")

        if from_file:
            return

        # Initialize content list; each method that adds any content
        # to the passband file needs to add a corresponding label to the
        # content list.
        self.content = []

        # Basic passband properties:
        self.pbset = pbset
        self.pbname = pbname
        self.calibrated = calibrated
        self.reference = reference
        self.version = version

        # Passband comments and history entries:
        self.history = []
        self.comments = []

        # Initialize an empty timestamp. This will get set by calling the save() method.
        self.timestamp = None

        # Passband transmission function table:
        if isinstance(ptf, str):
            ptf_table = np.loadtxt(ptf).T
            ptf_table[0] = ptf_table[0]*wlunits.to(u.m)
            self.ptf_table = {'wl': np.array(ptf_table[0]), 'fl': np.array(ptf_table[1])}
        elif isinstance(ptf, np.ndarray):
            self.ptf_table = {'wl': ptf[:,0]*wlunits.to(u.m), 'fl': ptf[:,1]}
        else:
            raise ValueError('argument `ptf` must either be a string (filename) or a (N, 2)-shaped array.')

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

        # Effective wavelength in wlunits:
        self.effwl = (self.ptf_photon_area/self.ptf_area*u.m).to(wlunits)

        # If any comments are passed, add them to history:
        if comment:
            self.add_comment(comment)

        self.add_to_history(f'{self.pbset}:{self.pbname} passband initialized.')

        # Initialize n-dimensional interpolators:
        self.ndp = dict()

    def __repr__(self):
        return f'<Passband: {self.pbset}:{self.pbname}>'

    def __str__(self):
        # old passband files do not have versions embedded, that is why we have to do this:
        if not hasattr(self, 'version') or self.version is None:
            self.version = 1.0
        return f'Passband: {self.pbset}:{self.pbname}\nVersion:  {self.version:1.1f}\nProvides: {self.content}\nHistory:  {self.history}'

    @property
    def log(self):
        h = f'{self.pbset}:{self.pbname} {self.version}\n'
        for entry in self.history:
            h += f'  {entry}\n'
        return h

    def add_to_history(self, history, max_length=46):
        """
        Adds a history entry to the passband file header.

        Parameters
        ----------
        * `comment` (string, required): comment to be added to the passband header.
        """

        if not isinstance(history, str):
            raise ValueError('passband header history entries must be strings.')
        if len(history) > max_length:
            raise ValueError(f'comment length should not exceed {max_length} characters.')

        self.history.append(f'{time.ctime()}: {history}')

    def add_comment(self, comment):
        """
        Adds a comment to the passband file header.

        Parameters
        ----------
        * `comment` (string, required): comment to be added to the passband header.
        """

        if not isinstance(comment, str):
            raise ValueError('passband header comments must be strings.')

        self.comments.append(comment)

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

        self.add_to_history(f'passband transmission function updated.')

    def save(self, archive, overwrite=True, update_timestamp=True, export_inorm_tables=False, export_legacy_comments=True):
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
        * `export_legacy_comments` (bool, optional, default=True): whether to
            export the dummy COMMENTS card (used in old passbands) in addition
            to the new COMMENT card.
        """

        # Timestamp is used for passband versioning.
        timestamp = time.ctime() if update_timestamp else self.timestamp

        header = fits.Header()
        header['PHOEBEVN'] = phoebe_version
        header['TIMESTMP'] = timestamp
        header['PBSET'] = self.pbset
        header['PBNAME'] = self.pbname
        header['EFFWL'] = self.effwl.value
        header['CALIBRTD'] = self.calibrated
        header['WLOVSMPL'] = self.wl_oversampling
        header['VERSION'] = self.version
        header['REFERENC'] = self.reference
        header['PTFORDER'] = self.ptf_order
        header['PTFEAREA'] = self.ptf_area
        header['PTFPAREA'] = self.ptf_photon_area

        if export_inorm_tables:
            self.content += [f'{atm}:Inorm' for atm in atm_tables.keys() if f'{atm}:Imu' in self.content]

        header['CONTENT'] = str(self.content)

        if export_legacy_comments:
            header['COMMENTS'] = ''

        # Add history entries:
        for entry in self.history:
            header['history'] = entry

        # Add comments:
        for comment in self.comments:
            header['comment'] = comment

        if 'extern_planckint:Inorm' in self.content or 'extern_atmx:Inorm' in self.content:
            header['WD_IDX'] = self.extern_wd_idx

        data = []

        primary_hdu = fits.PrimaryHDU(header=header)
        data.append(primary_hdu)

        data.append(fits.table_to_hdu(Table(self.ptf_table, meta={'extname': 'PTFTABLE'})))

        if 'blackbody:Inorm' in self.content:
            bb_func = Table({
                'teff': self._bb_func_energy[0],
                'logi_e': self._bb_func_energy[1],
                'logi_p': self._bb_func_photon[1]},
                meta={'extname': 'BB_FUNC'}
            )
            data.append(fits.table_to_hdu(bb_func))

        if 'blackbody:ext' in self.content:
            # concatenate basic and associated axes:
            axes = self.ndp['blackbody'].axes + self.ndp['blackbody'].table['ext@photon'][0]
            data.append(fits.table_to_hdu(Table({'teff': axes[0]}, meta={'extname': 'BB_TEFFS'})))
            data.append(fits.table_to_hdu(Table({'ebv': axes[1]}, meta={'extname': 'BB_EBVS'})))
            data.append(fits.table_to_hdu(Table({'rv': axes[2]}, meta={'extname': 'BB_RVS'})))

        # axes:
        for atm, prefix in atm_tables.items():
            if f'{atm}:Imu' in self.content:
                teffs, loggs, abuns, mus = self.ndp[atm].axes + self.ndp[atm].table['imu@photon'][0]
                data.append(fits.table_to_hdu(Table({'teff': teffs}, meta={'extname': f'{prefix}_TEFFS'})))
                data.append(fits.table_to_hdu(Table({'logg': loggs}, meta={'extname': f'{prefix}_LOGGS'})))
                data.append(fits.table_to_hdu(Table({'abun': abuns}, meta={'extname': f'{prefix}_ABUNS'})))
                data.append(fits.table_to_hdu(Table({'mu': mus}, meta={'extname': f'{prefix}_MUS'})))

                if f'{atm}:ext' in self.content:
                    ebvs, rvs = self.ndp[atm].table['ext@photon'][0]
                    data.append(fits.table_to_hdu(Table({'ebv': ebvs}, meta={'extname': f'{prefix}_EBVS'})))
                    data.append(fits.table_to_hdu(Table({'rv': rvs}, meta={'extname': f'{prefix}_RVS'})))

        # grids:
        if 'blackbody:ext' in self.content:
            data.append(fits.ImageHDU(self.ndp['blackbody'].table['ext@energy'][1], name='BBEGRID'))
            data.append(fits.ImageHDU(self.ndp['blackbody'].table['ext@photon'][1], name='BBPGRID'))

        for atm, prefix in atm_tables.items():
            if f'{atm}:Imu' in self.content:
                data.append(fits.ImageHDU(self.ndp[atm].table['imu@energy'][1], name=f'{prefix}FEGRID'))
                data.append(fits.ImageHDU(self.ndp[atm].table['imu@photon'][1], name=f'{prefix}FPGRID'))

                if export_inorm_tables:
                    data.append(fits.ImageHDU(self.ndp[atm].table['imu@energy'][1][..., -1, :], name=f'{prefix}NEGRID'))
                    data.append(fits.ImageHDU(self.ndp[atm].table['imu@photon'][1][..., -1, :], name=f'{prefix}NPGRID'))

            if f'{atm}:ld' in self.content:
                data.append(fits.ImageHDU(self.ndp[atm].table['ld@energy'][1], name=f'{prefix}LEGRID'))
                data.append(fits.ImageHDU(self.ndp[atm].table['ld@photon'][1], name=f'{prefix}LPGRID'))

            if f'{atm}:ldint' in self.content:
                data.append(fits.ImageHDU(self.ndp[atm].table['ldint@energy'][1], name=f'{prefix}IEGRID'))
                data.append(fits.ImageHDU(self.ndp[atm].table['ldint@photon'][1], name=f'{prefix}IPGRID'))

            if f'{atm}:ext' in self.content:
                data.append(fits.ImageHDU(self.ndp[atm].table['ext@energy'][1], name=f'{prefix}XEGRID'))
                data.append(fits.ImageHDU(self.ndp[atm].table['ext@photon'][1], name=f'{prefix}XPGRID'))

        pb = fits.HDUList(data)
        pb.writeto(archive, overwrite=overwrite)

    @classmethod
    def load(cls, archive, load_content=True, init_extrapolation=True):
        """
        Loads the passband contents from a fits file.

        Arguments
        ----------
        * `archive` (string): filename of the passband (in FITS format)
        * `load_content` (bool, optional, default=True): whether to load all
            table contents.  If False, only the headers will be loaded into
            the structure.
        * `init_extrapolation` (bool, optional, default=True): whether to
            initialize all structures needed for blending and interpolation.
            These are quite expensive to initialize.

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
            self.effwl = header['effwl'] * u.m
            self.calibrated = header['calibrtd']
            self.wl_oversampling = header.get('wlovsmpl', 1)
            self.reference = header['referenc']
            self.ptf_order = header['ptforder']
            self.ptf_area = header['ptfearea']
            self.ptf_photon_area = header['ptfparea']

            self.content = eval(header['content'], {'__builtins__': None}, {})

            self.history = list(header.get('history', ''))
            self.comments = list(header.get('comment', ''))

            # Initialize an ndpolator instance to hold all data:
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
                    self._log10_Inorm_bb_energy = lambda Teff: interpolate.splev(Teff, self._bb_func_energy).reshape(-1, 1)
                    self._log10_Inorm_bb_photon = lambda Teff: interpolate.splev(Teff, self._bb_func_photon).reshape(-1, 1)

                if 'blackbody:ext' in self.content:
                    axes = (
                        np.array(list(hdul['bb_teffs'].data['teff'])),
                        np.array(list(hdul['bb_ebvs'].data['ebv'])),
                        np.array(list(hdul['bb_rvs'].data['rv']))
                    )

                    self.ndp['blackbody'] = ndpolator.Ndpolator(basic_axes=(axes[0],))
                    self.ndp['blackbody'].register('ext@photon', (axes[1], axes[2]), hdul['bbegrid'].data)
                    self.ndp['blackbody'].register('ext@energy', (axes[1], axes[2]), hdul['bbpgrid'].data)

                for atm, prefix in atm_tables.items():
                    if f'{atm}:Imu' in self.content:
                        basic_axes = (
                            np.array(list(hdul[f'{prefix}_teffs'].data['teff'])),
                            np.array(list(hdul[f'{prefix}_loggs'].data['logg'])),
                            np.array(list(hdul[f'{prefix}_abuns'].data['abun'])),
                        )
                        mus = np.array(list(hdul[f'{prefix}_mus'].data['mu']))

                        atm_energy_grid = hdul[f'{prefix}fegrid'].data
                        atm_photon_grid = hdul[f'{prefix}fpgrid'].data

                        # ndpolator instance for interpolating and extrapolating:
                        self.ndp[atm] = ndpolator.Ndpolator(basic_axes=basic_axes)

                        # normal passband intensities:
                        self.ndp[atm].register('inorm@photon', None, atm_photon_grid[...,-1,:])
                        self.ndp[atm].register('inorm@energy', None, atm_energy_grid[...,-1,:])

                        # specific passband intensities:
                        self.ndp[atm].register('imu@photon', (mus,), atm_photon_grid)
                        self.ndp[atm].register('imu@energy', (mus,), atm_energy_grid)

                    if f'{atm}:ld' in self.content:
                        self.ndp[atm].register('ld@photon', None, hdul[f'{prefix}legrid'].data)
                        self.ndp[atm].register('ld@energy', None, hdul[f'{prefix}lpgrid'].data)

                    if f'{atm}:ldint' in self.content:
                        self.ndp[atm].register('ldint@photon', None, hdul[f'{prefix}iegrid'].data)
                        self.ndp[atm].register('ldint@energy', None, hdul[f'{prefix}ipgrid'].data)

                    if f'{atm}:ext' in self.content:
                        ebvs = np.array(list(hdul[f'{prefix}_ebvs'].data['ebv']))
                        rvs = np.array(list(hdul[f'{prefix}_rvs'].data['rv']))

                        self.ndp[atm].register('ext@photon', (ebvs, rvs), hdul[f'{prefix}XEGRID'].data)
                        self.ndp[atm].register('ext@energy', (ebvs, rvs), hdul[f'{prefix}XPGRID'].data)

        return self

    def _planck(self, lam, teff):
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

        return 2*h.value*c.value*c.value/lam**5 * 1./(np.exp(h.value*c.value/lam/k_B.value/teff)-1)

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

        expterm = np.exp(h.value*c.value/lam/k_B.value/Teff)
        return 2*h.value*c.value*c.value/k_B.value/Teff/lam**7 * (expterm-1)**-2 * (h.value*c.value*expterm-5*lam*k_B.value*Teff*(expterm-1))

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

        hclkt = h.value*c.value/lam/k_B.value/Teff
        expterm = np.exp(hclkt)
        return hclkt * expterm/(expterm-1)

    def compute_blackbody_intensities(self, teffs=None, include_extinction=False, rvs=None, ebvs=None, verbose=False):
        r"""
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
        pfs = 2*h.value*c.value*c.value/wls**5*1./(np.exp(h.value*c.value/(k_B.value*wls@teffs.reshape(1, -1)))-1)  # (47, 97)

        # Passband-weighted Planck functions:
        pbpfs_energy = self.ptf(wls).reshape(-1, 1)*pfs  # (47, 97)
        pbints_energy = np.log10(np.trapz(pbpfs_energy, self.wl, axis=0)/self.ptf_area)
        self._bb_func_energy = interpolate.splrep(teffs, pbints_energy, s=0)
        self._log10_Inorm_bb_energy = lambda teff: interpolate.splev(teff, self._bb_func_energy).reshape(-1, 1)

        pbpfs_photon = wls*self.ptf(wls).reshape(-1, 1)*pfs  # (47, 97)
        pbints_photon = np.log10(np.trapz(pbpfs_photon, self.wl, axis=0)/self.ptf_photon_area)
        self._bb_func_photon = interpolate.splrep(teffs, pbints_photon, s=0)
        self._log10_Inorm_bb_photon = lambda teff: interpolate.splev(teff, self._bb_func_photon).reshape(-1, 1)

        if 'blackbody:Inorm' not in self.content:
            self.content.append('blackbody:Inorm')

        if include_extinction:
            if ebvs is None:
                ebvs = np.linspace(0., 3., 30)
            if rvs is None:
                rvs = np.linspace(2., 6., 16)

            axes = (np.unique(teffs), np.unique(ebvs), np.unique(rvs))

            axbx = libphoebe.gordon_extinction(self.wl)
            ax, bx = axbx[:,0], axbx[:,1]

            # The following code broadcasts arrays so that integration can be vectorized:
            bb_sed = self._planck(self.wl[:, None], teffs[None, :])  # (54, 97)
            ptf = self.ptf(self.wl)[:, None]  # (54, 1)
            Alam = 10**(-0.4 * ebvs[None, :, None] * (rvs[None, None, :] * ax[:, None, None] + bx[:, None, None]))  # Shape (54, 30, 16)

            egrid = np.trapz(ptf[:, :, None, None] * bb_sed[:, :, None, None] * Alam[:, None, :, :], self.wl, axis=0) / np.trapz(ptf[:, :, None, None] * bb_sed[:, :, None, None], self.wl, axis=0)
            pgrid = np.trapz(self.wl[:, None, None, None] * ptf[:, :, None, None] * bb_sed[:, :, None, None] * Alam[:, None, :, :], self.wl, axis=0) / np.trapz(self.wl[:, None, None, None] * ptf[:, :, None, None] * bb_sed[:, :, None, None], self.wl, axis=0)

            self.ndp['blackbody'] = ndpolator.Ndpolator(basic_axes=(axes[0],))
            self.ndp['blackbody'].register('ext@photon', associated_axes=(axes[1], axes[2]), grid=pgrid[..., None])
            self.ndp['blackbody'].register('ext@energy', associated_axes=(axes[1], axes[2]), grid=egrid[..., None])

            if 'blackbody:ext' not in self.content:
                self.content.append('blackbody:ext')

            if verbose:
                print('')

        self.add_to_history(f"blackbody intensities {'with' if include_extinction else 'w/o'} extinction added.")

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
        * `units` (float): conversion units from model atmosphere intensity
          units to W/m^3.
        """

        models = glob.glob(path+'/*fits')
        nmodels = len(models)
        teffs, loggs, abuns = np.empty(nmodels), np.empty(nmodels), np.empty(nmodels)

        if atm == 'ck2004':
            mus = np.array([0., 0.001, 0.002, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])
            wls = np.arange(900., 39999.501, 0.5)/1e10  # AA -> m
            for i, model in enumerate(models):
                relative_filename = model[model.rfind('/')+1:] # get relative pathname
                teffs[i] = float(relative_filename[1:6])
                loggs[i] = float(relative_filename[7:9])/10
                abuns[i] = float(relative_filename[10:12])/10 * (-1 if relative_filename[9] == 'M' else 1)
            units = 1e7  # erg/s/cm^2/A -> W/m^3
        elif atm == 'phoenix':
            mus = np.array([0., 0.001, 0.002, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])
            wls = np.arange(500., 26000.)/1e10  # AA -> m
            for i, model in enumerate(models):
                relative_filename = model[model.rfind('/')+1:] # get relative pathname
                teffs[i] = float(relative_filename[1:6])
                loggs[i] = float(relative_filename[7:11])
                abuns[i] = float(relative_filename[12:16])
            units = 1  # W/m^3
        elif atm in ['tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO']:
            mus = np.array([0., 0.00136799, 0.00719419, 0.01761889, 0.03254691, 0.05183939, 0.07531619, 0.10275816, 0.13390887, 0.16847785, 0.20614219, 0.24655013, 0.28932435, 0.33406564, 0.38035639, 0.42776398, 0.47584619, 0.52415388, 0.57223605, 0.6196437, 0.66593427, 0.71067559, 0.75344991, 0.79385786, 0.83152216, 0.86609102, 0.89724188, 0.92468378, 0.9481606,  0.96745302, 0.98238112, 0.99280576, 0.99863193, 1.])
            wls = np.load(path+'/wavelengths.npy')  # in meters
            for i, model in enumerate(models):
                pars = re.split('[TGA.]+', model[model.rfind('/')+1:])
                teffs[i] = float(pars[1])
                loggs[i] = float(pars[2])/100
                abuns[i] = float(pars[3])/100
            units = 1  # W/m^3
        else:
            raise ValueError(f'atm={atm} is not supported.')

        return models, teffs, loggs, abuns, mus, wls, units

    def compute_intensities(self, atm, path, include_extinction=False, rvs=None, ebvs=None, verbose=True):
        """
        Computes direction-dependent passband intensities using the passed `atm`
        model atmospheres.

        Arguments
        ----------
        * `atm` (string): name of the model atmosphere
        * `path` (string): path to the directory with SEDs in FITS format.
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

        models, teffs, loggs, abuns, mus, wls, units = self.parse_atm_datafiles(atm, path)
        nmodels = len(models)

        ints_energy, ints_photon = np.empty(nmodels*len(mus)), np.empty(nmodels*len(mus))

        keep = (wls >= self.ptf_table['wl'][0]) & (wls <= self.ptf_table['wl'][-1])
        wls = wls[keep]
        ptf = self.ptf(wls)

        if include_extinction:
            if ebvs is None:
                ebvs = np.linspace(0., 3., 30)
            if rvs is None:
                rvs = np.linspace(2., 6., 16)

            ext_axes = (np.unique(teffs), np.unique(loggs), np.unique(abuns), ebvs, rvs)
            ext_photon_grid = np.empty(shape=(len(np.unique(teffs)), len(np.unique(loggs)), len(np.unique(abuns)), len(ebvs), len(rvs), 1))
            ext_energy_grid = np.empty(shape=(len(np.unique(teffs)), len(np.unique(loggs)), len(np.unique(abuns)), len(ebvs), len(rvs), 1))

            axbx = libphoebe.gordon_extinction(wls)
            ax, bx = axbx[:,0], axbx[:,1]
            
            # The following code broadcasts arrays so that integration can be vectorized:
            Alam = 10**(-0.4 * ebvs[None, :, None] * (rvs[None, None, :] * ax[:, None, None] + bx[:, None, None]))

        for i, model in tqdm(enumerate(models), desc=atm, total=len(models), disable=not verbose, unit=' models'):
            with fits.open(model) as hdu:
                seds = hdu[0].data*units  # must be in in W/m^3

                # trim intensities to the passband limits:
                seds = seds[:,keep]

                pbints_energy = ptf*seds
                fluxes_energy = np.trapz(pbints_energy, wls)

                pbints_photon = wls*pbints_energy
                fluxes_photon = np.trapz(pbints_photon, wls)

                # work around log10(flux(mu=0)=0) = -inf:
                fluxes_energy[mus < 1e-12] = fluxes_photon[mus < 1e-12] = 1

                ints_energy[i*len(mus):(i+1)*len(mus)] = np.log10(fluxes_energy/self.ptf_area)         # energy-weighted intensity
                ints_photon[i*len(mus):(i+1)*len(mus)] = np.log10(fluxes_photon/self.ptf_photon_area)  # photon-weighted intensity

                if include_extinction:
                    # we only use normal emergent intensities here for simplicity:
                    epbints = pbints_energy[-1].reshape(-1, 1)
                    egrid = np.trapz(epbints[:, :, None, None] * Alam[:, None, :, :], wls, axis=0) / np.trapz(epbints[:, :, None, None], wls, axis=0)

                    ppbints = pbints_photon[-1].reshape(-1, 1)
                    pgrid = np.trapz(ppbints[:, :, None, None] * Alam[:, None, :, :], wls, axis=0) / np.trapz(ppbints[:, :, None, None], wls, axis=0)

                    t = (teffs[i] == ext_axes[0], loggs[i] == ext_axes[1], abuns[i] == ext_axes[2])
                    ext_energy_grid[t] = egrid.reshape(len(ebvs), len(rvs), 1)
                    ext_photon_grid[t] = pgrid.reshape(len(ebvs), len(rvs), 1)

        basic_axes = (np.unique(teffs), np.unique(loggs), np.unique(abuns))
        self.ndp[atm] = ndpolator.Ndpolator(basic_axes=basic_axes)

        associated_axes = (np.unique(mus),)
        axes = basic_axes + associated_axes

        atm_energy_grid = np.full(shape=[len(axis) for axis in axes]+[1], fill_value=np.nan)
        atm_photon_grid = np.copy(atm_energy_grid)

        for i, int_energy in enumerate(ints_energy):
            atm_energy_grid[teffs[int(i/len(mus))] == axes[0], loggs[int(i/len(mus))] == axes[1], abuns[int(i/len(mus))] == axes[2], mus[i%len(mus)] == axes[3], 0] = int_energy
        for i, int_photon in enumerate(ints_photon):
            atm_photon_grid[teffs[int(i/len(mus))] == axes[0], loggs[int(i/len(mus))] == axes[1], abuns[int(i/len(mus))] == axes[2], mus[i%len(mus)] == axes[3], 0] = int_photon

        self.ndp[atm].register('inorm@photon', None, atm_photon_grid[...,-1,:])
        self.ndp[atm].register('inorm@energy', None, atm_energy_grid[...,-1,:])
        self.ndp[atm].register('imu@photon', associated_axes, atm_photon_grid)
        self.ndp[atm].register('imu@energy', associated_axes, atm_energy_grid)

        if f'{atm}:Imu' not in self.content:
            self.content.append(f'{atm}:Imu')

        if include_extinction:
            associated_axes = (np.unique(ebvs), np.unique(rvs))
            axes = basic_axes + associated_axes

            self.ndp[atm].register('ext@photon', associated_axes, ext_photon_grid)
            self.ndp[atm].register('ext@energy', associated_axes, ext_energy_grid)

            if f'{atm}:ext' not in self.content:
                self.content.append(f'{atm}:ext')

        self.add_to_history(f"{atm} intensities {'with' if include_extinction else 'w/o'} extinction added.")

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

    def compute_ldcoeffs(self, ldatm, weighting='uniform'):
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
        """

        if f'{ldatm}:Imu' not in self.content:
            raise RuntimeError(f'atm={ldatm} intensities are not found in the {self.pbset}:{self.pbname} passband.')

        basic_axes = self.ndp[ldatm].axes
        mus = self.ndp[ldatm].table['imu@photon'][0][0]
        if ldatm[:4] == 'tmap':
            # remove extrapolated points in mu for TMAP family of model atmospheres:
            mus = mus[1:-1]

        ld_energy_grid = np.full(shape=[len(axis) for axis in basic_axes]+[11], fill_value=np.nan)
        ld_photon_grid = np.copy(ld_energy_grid)

        if weighting == 'uniform':
            sigma = np.ones(len(mus))
        elif weighting == 'interval':
            delta = np.concatenate( (np.array((mus[1]-mus[0],)), mus[1:]-mus[:-1]) )
            sigma = 1./np.sqrt(delta)
        else:
            raise ValueError(f'weighting={weighting} is not supported.')

        atm_energy_grid = self.ndp[ldatm].table['imu@energy'][1]
        atm_photon_grid = self.ndp[ldatm].table['imu@photon'][1]

        for Tindex in range(len(basic_axes[0])):
            for lindex in range(len(basic_axes[1])):
                for mindex in range(len(basic_axes[2])):
                    if ldatm[:4] == 'tmap':
                        IsE = 10**atm_energy_grid[Tindex,lindex,mindex,1:-1].flatten()
                    else:
                        IsE = 10**atm_energy_grid[Tindex,lindex,mindex,:].flatten()
                    fEmask = np.isfinite(IsE)
                    if len(IsE[fEmask]) <= 1:
                        continue
                    IsE /= IsE[fEmask][-1]

                    cElin,  pcov = cfit(f=self._ldlaw_lin,    xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5])
                    cElog,  pcov = cfit(f=self._ldlaw_log,    xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cEsqrt, pcov = cfit(f=self._ldlaw_sqrt,   xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cEquad, pcov = cfit(f=self._ldlaw_quad,   xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cEnlin, pcov = cfit(f=self._ldlaw_nonlin, xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma[fEmask], p0=[0.5, 0.5, 0.5, 0.5])
                    ld_energy_grid[Tindex, lindex, mindex] = np.hstack((cElin, cElog, cEsqrt, cEquad, cEnlin))

                    if ldatm[:4] == 'tmap':
                        IsP = 10**atm_photon_grid[Tindex,lindex,mindex,1:-1].flatten()
                    else:
                        IsP = 10**atm_photon_grid[Tindex,lindex,mindex,:].flatten()
                    fPmask = np.isfinite(IsP)
                    IsP /= IsP[fPmask][-1]

                    cPlin,  pcov = cfit(f=self._ldlaw_lin,    xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5])
                    cPlog,  pcov = cfit(f=self._ldlaw_log,    xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cPsqrt, pcov = cfit(f=self._ldlaw_sqrt,   xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cPquad, pcov = cfit(f=self._ldlaw_quad,   xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5, 0.5])
                    cPnlin, pcov = cfit(f=self._ldlaw_nonlin, xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma[fEmask], p0=[0.5, 0.5, 0.5, 0.5])
                    ld_photon_grid[Tindex, lindex, mindex] = np.hstack((cPlin, cPlog, cPsqrt, cPquad, cPnlin))

        self.ndp[ldatm].register('ld@photon', None, ld_photon_grid)
        self.ndp[ldatm].register('ld@energy', None, ld_energy_grid)

        if f'{ldatm}:ld' not in self.content:
            self.content.append(f'{ldatm}:ld')

        self.add_to_history(f'LD coefficients for {ldatm} added.')

    def export_phoenix_atmtab(self):
        """
        Exports PHOENIX intensity table to a PHOEBE legacy compatible format.
        """

        teffs = self.ndp['phoenix'].axes[0]
        tlow, tup = teffs[0], teffs[-1]
        trel = (teffs-tlow)/(tup-tlow)

        for abun in range(len(self.ndp['phoenix'].axes[2])):
            for logg in range(len(self.ndp['phoenix'].axes[1])):
                logI = self.ndp['phoenix'].table['imu@energy'][1][:,logg,abun,-1,0]+1 # +1 to take care of WD units

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
        Exports  limb darkening coefficients to a PHOEBE legacy compatible format.

        Arguments
        -----------
        * `models` (string): the path (including the filename) of legacy's
            models.list
        * `atm` (string, default='ck2004'): atmosphere model, 'ck2004' or 'phoenix'
        * `filename` (string, optional, default=None): output filename for
            storing the table
        * `intens_weighting`
        """

        axes = self.ndp[atm].axes
        grid = self.ndp[atm].table['ld@photon'][1] if intens_weighting == 'photon' else self.ndp[atm].table['ld@energy'][1]

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
        r"""
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

        ldaxes = self.ndp[ldatm].axes
        ldtable = self.ndp[ldatm].table['imu@energy'][1]
        pldtable = self.ndp[ldatm].table['imu@photon'][1]

        ldint_energy_grid = np.nan*np.ones((len(ldaxes[0]), len(ldaxes[1]), len(ldaxes[2]), 1))
        ldint_photon_grid = np.nan*np.ones((len(ldaxes[0]), len(ldaxes[1]), len(ldaxes[2]), 1))

        mu = self.ndp[ldatm].table['imu@photon'][0][0]
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

                    ldint_energy_grid[a,b,c] = 2*ldint
                    ldint_photon_grid[a,b,c] = 2*pldint

        self.ndp[ldatm].register('ldint@photon', None, ldint_photon_grid)
        self.ndp[ldatm].register('ldint@energy', None, ldint_energy_grid)

        if f'{ldatm}:ldint' not in self.content:
            self.content.append(f'{ldatm}:ldint')

        self.add_to_history(f'LD integrals for {ldatm} added.')

    def interpolate_ldcoeffs(self, query_pts, ldatm='ck2004', ld_func='power', intens_weighting='photon', ld_extrapolation_method='none'):
        """
        Interpolate the passband-stored table of LD model coefficients.

        Arguments
        ------------
        * `query_pts` (ndarray, required): a C-contiguous DxN array of queried points
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

        ld_coeffs = self.ndp[ldatm].ndpolate(f'ld@{intens_weighting}', query_pts, extrapolation_method=ld_extrapolation_method)['interps']
        return ld_coeffs[s[ld_func]]

    def interpolate_extinct(self, query_pts, atm='blackbody', intens_weighting='photon', extrapolation_method='none'):
        """
        Interpolates the passband-stored tables of extinction corrections

        Arguments
        ----------
        * `query_pts` (ndarray): an NxD-dimensional ndarray, where N is the number of query points and D their dimension
        * `atm` (string, optional, default='blackbody'): atmosphere model.
        * `intens_weighting`
        * `extrapolation_method`

        Returns
        ---------
        * extinction factor

        Raises
        --------
        * ValueError if `atm` is not supported.
        """

        if f'{atm}:ext' not in self.content:
            raise ValueError(f"extinction factors for atm={atm} not found for the {self.pbset}:{self.pbname} passband.")

        ndp = self.ndp[atm]
        if atm == 'blackbody':
            # if atm == 'blackbody', we need to remove any excess columns from
            # query points.
            reduced_query_pts = np.ascontiguousarray(query_pts[:, [0, -2, -1]])
            extinct_factor = ndp.ndpolate(f'ext@{intens_weighting}', reduced_query_pts, extrapolation_method=extrapolation_method)['interps']
        else:
            extinct_factor = ndp.ndpolate(f'ext@{intens_weighting}', query_pts, extrapolation_method=extrapolation_method)['interps']

        # print(f'{reduced_query_pts=} {extinct_factor=}')
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

        if wdidx < 1 or wdidx > Npb:
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

        self.add_to_history(f'Wilson-Devinney atmosphere tables imported.')

    def _log10_Inorm_extern_planckint(self, teffs):
        """
        Internal function to compute normal passband intensities using
        the external WD machinery that employs blackbody approximation.

        @teffs: effective temperature in K

        Returns: log10(Inorm)
        """

        log10_Inorm = libphoebe.wd_planckint(teffs, self.extern_wd_idx, self.wd_data["planck_table"])

        return log10_Inorm.reshape(-1, 1)

    def _log10_Inorm_extern_atmx(self, query_pts):
        """
        Internal function to compute normal passband intensities using
        the external WD machinery that employs model atmospheres and
        ramps.

        Arguments
        ----------
        * `query_pts` (ndarray, required): a C-contiguous DxN array of queried points

        Returns
        ----------
        * log10(Inorm)
        """

        log10_Inorm = libphoebe.wd_atmint(
            np.ascontiguousarray(query_pts[:,0]),  # teff
            np.ascontiguousarray(query_pts[:,1]),  # logg
            np.ascontiguousarray(query_pts[:,2]),  # abun
            self.extern_wd_idx,
            self.wd_data["planck_table"],
            self.wd_data["atm_table"]
        ) - 1  # -1 for cgs -> metric

        return log10_Inorm.reshape(-1, 1)

    def _log10_Inorm(self, query_pts, atm, intens_weighting='photon', atm_extrapolation_method='none', ld_extrapolation_method='none', blending_method='none', raise_on_nans=True, return_nanmask=False):
        """
        Computes normal emergent passband intensities for model atmospheres.

        Parameters
        ----------
        * `query_pts` (ndarray, required): a C-contiguous DxN array of queried points
        atm : string
            model atmosphere ('ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO')
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

        ndpolants = self.ndp[atm].ndpolate(f'inorm@{intens_weighting}', query_pts, extrapolation_method=atm_extrapolation_method)
        log10_Inorm = ndpolants['interps']
        dists = ndpolants['dists']

        offgrid = dists > 1e-5
        if not np.any(offgrid):
            return ndpolants

        return log10_Inorm

    def Inorm(self, query_pts, atm='ck2004', ldatm='ck2004', ldint=None, ld_func='interp', ld_coeffs=None, intens_weighting='photon', atm_extrapolation_method='none', ld_extrapolation_method='none', blending_method='none', blending_margin=3, dist_threshold=1e-5):
        r"""
        Computes normal emergent passband intensity.

        Possible atm/ldatm/ld_func/ld_coeffs combinations:

        | atm       | ldatm         | ld_func                 | ld_coeffs | intens_weighting | action                                                      |
        ------------|---------------|-------------------------|-----------|------------------|-------------------------------------------------------------|
        | blackbody | none          | *                       | none      | *                | raise error                                                 |
        | blackbody | none          | lin,log,quad,sqrt,power | *         | *                | use manual LD model                                         |
        | blackbody | supported atm | interp                  | none      | *                | interpolate from ldatm                                      |
        | blackbody | supported atm | interp                  | *         | *                | interpolate from ldatm but warn about unused ld_coeffs      |
        | blackbody | supported atm | lin,log,quad,sqrt,power | none      | *                | interpolate ld_coeffs from ck2004:ld                        |
        | blackbody | supported atm | lin,log,quad,sqrt,power | *         | *                | use manual LD model but warn about unused ldatm             |
        | planckint | *             | *                       | *         | photon           | raise error                                                 |
        | atmx      | *             | *                       | *         | photon           | raise error                                                 |
        | ck2004    |               |                         |           |                  |                                                             |
        | phoenix   |               |                         |           |                  |                                                             |
        | tmap      |               |                         |           |                  |                                                             |

        Arguments
        ----------
        * `query_pts` (ndarray, required): a C-contiguous DxN array of queried points
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
        * `dist_threshold` (float, optional, default=1e-5): off-grid distance
          threshold. Query points farther than this value, in hypercube-
          normalized units, are considered off-grid.
        * `blending_margin` (float, optional, default=3): the off-grid region,
          in hypercube-normalized units, where blending should be done.

        Returns
        ----------
        * (dict) a dict of normal emergent passband intensities and associated
          values. Dictionary keys are: 'inorms' (required; normal intensities),
          'dists' (optional, distances from the grid); 'nanmask' (optional, a
          boolean mask where inorms are nan).

        Raises
        ----------
        * ValueError: if atmosphere parameters are out of bounds for the
          table.
        * NotImplementedError: if `ld_func` is not supported.
        """
        # if atm not in ['blackbody', 'extern_planckint', 'extern_atmx', 'ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO']:
        #     raise ValueError(f'atm={atm} is not supported.')

        # if ldatm not in ['none', 'ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO']:
        #     raise ValueError(f'ldatm={ldatm} is not supported.')

        # if intens_weighting not in ['energy', 'photon']:
        #     raise ValueError(f'intens_weighting={intens_weighting} is not supported.')

        # if blending_method not in ['none', 'blackbody']:
        #     raise ValueError(f'blending_method={blending_method} is not supported.')

        raise_on_nans = True if atm_extrapolation_method == 'none' else False
        blending_factors = None

        if atm == 'blackbody' and 'blackbody:Inorm' in self.content:
            # check if the required tables for the chosen ldatm are available:
            # if ldatm == 'none' and ld_coeffs is None:
            #     raise ValueError("ld_coeffs must be passed when ldatm='none'.")
            # if ld_func == 'interp' and f'{ldatm}:Imu' not in self.content:
            #     raise RuntimeError(f'passband {self.pbset}:{self.pbname} does not contain specific intensities for ldatm={ldatm}.')
            # if ld_func != 'interp' and ld_coeffs is None and f'{ldatm}:ld' not in self.content:
            #     raise RuntimeError(f'passband {self.pbset}:{self.pbname} does not contain limb darkening coefficients for ldatm={ldatm}.')
            # if blending_method == 'blackbody':
            #     raise ValueError(f'the combination of atm={atm} and blending_method={blending_method} is not valid.')

            if intens_weighting == 'photon':
                intensities = 10**self._log10_Inorm_bb_photon(query_pts[:,0])
            elif intens_weighting == 'energy':
                intensities = 10**self._log10_Inorm_bb_energy(query_pts[:,0])
            else:
                raise ValueError(f'{intens_weighting=} not recognized, must be "photon" or "energy".')

            if ldint is None:
                if ld_func != 'interp' and ld_coeffs is None:
                    ld_coeffs = self.interpolate_ldcoeffs(query_pts=query_pts, ldatm=ldatm, ld_func=ld_func, intens_weighting=intens_weighting, ld_extrapolation_method=ld_extrapolation_method)
                ldint = self.ldint(query_pts=query_pts, ldatm=ldatm, ld_func=ld_func, ld_coeffs=ld_coeffs, intens_weighting=intens_weighting, ld_extrapolation_method=ld_extrapolation_method, raise_on_nans=raise_on_nans)

            intensities /= ldint

        elif atm == 'extern_planckint' and 'extern_planckint:Inorm' in self.content:
            if intens_weighting == 'photon':
                raise ValueError(f'the combination of atm={atm} and intens_weighting={intens_weighting} is not supported.')
            # TODO: add all other exceptions

            intensities = 10**(self._log10_Inorm_extern_planckint(np.ascontiguousarray(query_pts[:,0]))-1)  # -1 is for cgs -> SI
            if ldint is None:
                ldint = self.ldint(query_pts=query_pts, ldatm=ldatm, ld_func=ld_func, ld_coeffs=ld_coeffs, intens_weighting=intens_weighting, ld_extrapolation_method=ld_extrapolation_method, raise_on_nans=raise_on_nans)
            
            # print(f'{intensities.shape=} {ldint.shape=} {intensities[:5]=} {ldint[:5]=}')
            intensities /= ldint

        elif atm == 'extern_atmx' and 'extern_atmx:Inorm' in self.content:
            if intens_weighting == 'photon':
                raise ValueError(f'the combination of atm={atm} and intens_weighting={intens_weighting} is not supported.')
            # TODO: add all other exceptions

            intensities = 10**(self._log10_Inorm_extern_atmx(query_pts=query_pts))

        else:  # atm in one of the model atmospheres
            if f'{atm}:Imu' not in self.content:
                raise ValueError(f'atm={atm} tables are not available in the {self.pbset}:{self.pbname} passband.')

            ndpolants = self.ndp[atm].ndpolate(f'inorm@{intens_weighting}', query_pts, extrapolation_method=atm_extrapolation_method)

            log10ints = ndpolants['interps']
            dists = ndpolants.get('dists', np.zeros_like(log10ints))

            if np.any(dists > dist_threshold) and blending_method == 'blackbody':
                ints_bb = self.Inorm(
                    query_pts=query_pts,
                    atm='blackbody',
                    ldatm=ldatm,
                    ldint=ldint,
                    ld_func=ld_func,
                    ld_coeffs=ld_coeffs,
                    intens_weighting=intens_weighting,
                    atm_extrapolation_method=atm_extrapolation_method,
                    ld_extrapolation_method=ld_extrapolation_method                    
                )
                log10ints_bb = np.log10(ints_bb['inorms'])

                off_grid = dists > dist_threshold

                log10ints_blended = log10ints.copy()
                log10ints_blended[off_grid] = (np.minimum(dists[off_grid], blending_margin) * log10ints_bb[off_grid] + np.maximum(blending_margin-dists[off_grid], 0) * log10ints[off_grid])/blending_margin
                blending_factors = np.minimum(dists, blending_margin)/blending_margin

                intensities = 10**log10ints_blended
            else:
                intensities = 10**log10ints

        ints = {
            'inorms': intensities,
            'bfs': blending_factors,
            # TODO: add any other dict keys?
        }

        return ints

    def _log10_Imu(self, atm, query_pts, intens_weighting='photon', atm_extrapolation_method='none', ld_extrapolation_method='none', blending_method='none', raise_on_nans=True):
        """
        Computes specific emergent passband intensities for model atmospheres.

        Parameters
        ----------
        atm : string
            model atmosphere ('ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO')
        * `query_pts` (ndarray, required): a C-contiguous DxN array of queried points
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

        Returns
        -------
        log10_Imu : dict
            keys: 'interps' (required), 'dists' (optional)
            interpolated (possibly extrapolated, blended) model atmosphre
            intensity

        Raises
        ------
        ValueError
            when interpolants are nan and raise_on_nans=True
        """

        ndpolants = self.ndp[atm].ndpolate(f'imu@{intens_weighting}', query_pts, extrapolation_method=atm_extrapolation_method)
        log10_Imu = ndpolants['interps']
        dists = ndpolants['dists']

        if raise_on_nans and np.any(dists > 1e-5):
            raise ValueError('specific intensity interpolation failed: queried atmosphere values are out of bounds.')

        nanmask = np.isnan(log10_Imu)
        if ~np.any(nanmask):
            return ndpolants

        if blending_method == 'blackbody':
            log10_Imu_bb = np.log10(self.Imu(query_pts=query_pts[nanmask], atm='blackbody', ldatm=atm, ld_extrapolation_method=ld_extrapolation_method, intens_weighting=intens_weighting))
            log10_Imu_blended = log10_Imu[:]
            log10_Imu_blended[nanmask] = np.min(dists[nanmask], 3)*log10_Imu_bb[nanmask] + np.max(3-dists[nanmask], 0)*log10_Imu[nanmask]
            return {'interps': log10_Imu_blended, 'dists': dists}

        return ndpolants

    def Imu(self, query_pts, atm='ck2004', ldatm='ck2004', ldint=None, ld_func='interp', ld_coeffs=None, intens_weighting='photon', atm_extrapolation_method='none', ld_extrapolation_method='none', blending_method='none', dist_threshold=1e-5, blending_margin=3):
        r"""
        Computes specific emergent passband intensities.

        Arguments
        ----------
        * `query_pts` (ndarray, required): a C-contiguous DxN array of queried points
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
        * `dist_threshold` (float, optional, default=1e-5): off-grid distance
          threshold. Query points farther than this value, in hypercube-
          normalized units, are considered off-grid.
        * `blending_margin` (float, optional, default=3): the off-grid region,
          in hypercube-normalized units, where blending should be done.


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

        if atm not in ['blackbody', 'extern_planckint', 'extern_atmx', 'ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO']:
            raise RuntimeError(f'atm={atm} is not supported.')

        if ldatm not in ['none', 'ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO']:
            raise ValueError(f'ldatm={ldatm} is not supported.')

        if ld_func == 'interp':
            if atm == 'blackbody' and 'blackbody:Inorm' in self.content and ldatm in ['ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO']:
                # we need to apply ldatm's limb darkening to blackbody intensities:
                #   Imu^bb = Lmu Inorm^bb = Imu^atm / Inorm^atm * Inorm^bb

                # print(f'{atm=} {ld_func=} {ldatm=} {intens_weighting=} {query_pts=} {atm_extrapolation_method=}')

                ndpolants = self.ndp[ldatm].ndpolate(f'imu@{intens_weighting}', query_pts, extrapolation_method=atm_extrapolation_method)
                log10imus_atm = ndpolants['interps']
                dists = ndpolants.get('dists', np.zeros_like(log10imus_atm))

                reduced_query_pts = query_pts[:,:-1]
                # print(f'{reduced_query_pts.shape=} {atm=} {ldatm=} {ldint=} {ld_func=} {ld_coeffs=} {intens_weighting=} {atm_extrapolation_method=} {ld_extrapolation_method=} {blending_method=}')

                ints_atm = self.Inorm(
                    query_pts=reduced_query_pts,
                    atm=ldatm,
                    ldatm=ldatm,
                    ldint=ldint,
                    ld_func=ld_func,
                    ld_coeffs=ld_coeffs,
                    intens_weighting=intens_weighting,
                    atm_extrapolation_method=atm_extrapolation_method,
                    ld_extrapolation_method=ld_extrapolation_method
                )
                log10inorms_atm = np.log10(ints_atm['inorms'])

                ints_bb = self.Inorm(
                    query_pts=reduced_query_pts,
                    atm='blackbody',
                    ldatm=ldatm,
                    ldint=ldint,
                    ld_func=ld_func,
                    ld_coeffs=ld_coeffs,
                    intens_weighting=intens_weighting,
                    atm_extrapolation_method=atm_extrapolation_method,
                    ld_extrapolation_method=ld_extrapolation_method                    
                )
                log10inorms_bb = np.log10(ints_bb['inorms'])

                log10imus_bb = log10imus_atm / log10inorms_atm * log10inorms_bb
                
                return 10**log10imus_bb
            
            elif atm == 'blackbody' and 'blackbody:Inorm' in self.content and ldatm not in ['ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO']:
                raise ValueError(f'{atm=} and {ld_func=} are incompatible with {ldatm=}.')

            elif atm in ['ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO']:
                if f'{atm}:Imu' not in self.content:
                    raise ValueError(f'{atm=} tables are not available in the {self.pbset}:{self.pbname} passband.')

                ndpolants = self.ndp[atm].ndpolate(f'imu@{intens_weighting}', query_pts, extrapolation_method=atm_extrapolation_method)
                log10imus_atm = ndpolants['interps']
                dists = ndpolants.get('dists', np.zeros_like(log10imus_atm))

                if np.any(dists > dist_threshold) and blending_method == 'blackbody':
                    off_grid = (dists > dist_threshold).flatten()
                    # print(f'{query_pts.shape=} {off_grid.shape=}')

                    ints_bb = self.Imu(
                        query_pts=query_pts[off_grid],
                        atm='blackbody',
                        ldatm=ldatm,
                        ldint=ldint,
                        ld_func=ld_func,
                        ld_coeffs=ld_coeffs,
                        intens_weighting=intens_weighting,
                        atm_extrapolation_method=atm_extrapolation_method,
                        ld_extrapolation_method=ld_extrapolation_method
                    )
                    log10imus_bb = np.log10(ints_bb)

                    log10imus_blended = log10imus_atm.copy()
                    log10imus_blended[off_grid] = (np.minimum(dists[off_grid], blending_margin) * log10imus_bb + np.maximum(blending_margin-dists[off_grid], 0) * log10imus_atm[off_grid])/blending_margin

                    intensities = 10**log10imus_blended
                else:
                    intensities = 10**log10imus_atm

                return intensities

            else:
                # anything else we need to special-handle for ld_func == 'interp'?
                pass

        else:  # if ld_func != 'interp':
            mus = query_pts[:,-1]
            reduced_query_pts = query_pts[:,:-1]
            
            # print(f'{query_pts=}, {mus=}, {atm=}, {ldatm=}, {ldint=}, {ld_func=}, {ld_coeffs=}, {intens_weighting=}, {atm_extrapolation_method=}, {ld_extrapolation_method=}, {blending_method=}, {return_nanmask=}')

            if ld_coeffs is None:
                # LD function can be passed without coefficients; in that
                # case we need to interpolate them from the tables.
                ld_coeffs = self.interpolate_ldcoeffs(reduced_query_pts, ldatm, ld_func, intens_weighting=intens_weighting, ld_extrapolation_method=ld_extrapolation_method)

            ints = self.Inorm(
                query_pts=reduced_query_pts,
                atm=atm,
                ldatm=ldatm,
                ldint=ldint,
                ld_func=ld_func,
                ld_coeffs=ld_coeffs,
                intens_weighting=intens_weighting,
                atm_extrapolation_method=atm_extrapolation_method,
                ld_extrapolation_method=ld_extrapolation_method,
                blending_method=blending_method
            )

            ld = self._ld(ld_func=ld_func, mu=mus, ld_coeffs=ld_coeffs).reshape(-1, 1)

            return ints['inorms'] * ld

    def ldint(self, query_pts, ldatm=None, ld_func='linear', ld_coeffs=np.array([[0.5]]), intens_weighting='photon', ld_extrapolation_method='none', raise_on_nans=True):
        """
        Computes ldint value for the given `ld_func` and `ld_coeffs`.

        Arguments
        ----------
        * `query_pts` (ndarray, required): a C-contiguous DxN array of queried points
        * `ldatm` (string, optional, default=None): limb darkening model atmosphere
        * `ld_func` (string, optional, default='linear'): limb darkening function
        * `ld_coeffs` (array, optional, default=[[0.5]]): limb darkening coefficients
        * `intens_weighting` (string, optional, default='photon'): intensity weighting mode
        * `ld_extrapolation_mode` (string, optional, default='none): extrapolation mode
        * `raise_on_nans` (boolean, optional, default=True): should any nans raise an exception

        Returns
        -------
        * (array) ldint value(s)
        """

        if ld_func == 'interp':
            ldints = self.ndp[ldatm].ndpolate(f'ldint@{intens_weighting}', query_pts, extrapolation_method=ld_extrapolation_method)['interps']
            return ldints

        if ld_coeffs is not None:
            ld_coeffs = np.atleast_2d(ld_coeffs)

        if ld_coeffs is None:
            ld_coeffs = self.interpolate_ldcoeffs(query_pts=query_pts, ldatm=ldatm, ld_func=ld_func, intens_weighting=intens_weighting, ld_extrapolation_method=ld_extrapolation_method)

        ldints = np.ones(shape=(len(query_pts), 1))

        if ld_func == 'linear':
            ldints[:,0] *= 1-ld_coeffs[:,0]/3
        elif ld_func == 'logarithmic':
            ldints[:,0] *= 1-ld_coeffs[:,0]/3+2.*ld_coeffs[:,1]/9
        elif ld_func == 'square_root':
            ldints[:,0] *= 1-ld_coeffs[:,0]/3-ld_coeffs[:,1]/5
        elif ld_func == 'quadratic':
            ldints[:,0] *= 1-ld_coeffs[:,0]/3-ld_coeffs[:,1]/6
        elif ld_func == 'power':
            ldints[:,0] *= 1-ld_coeffs[:,0]/5-ld_coeffs[:,1]/3-3.*ld_coeffs[:,2]/7-ld_coeffs[:,3]/2
        else:
            raise ValueError(f'ld_func={ld_func} is not recognized.')

        return ldints

    def _bindex_blackbody(self, Teff, intens_weighting='photon'):
        r"""
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
        bindex = libphoebe.interp(req, self.ndp['ck2004'].axes, grid).T[0]
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
    r"""
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

    try:
        pb = Passband.load('tables/passbands/bolometric.fits')
    except FileNotFoundError:
        pb = Passband(
            ptf='tables/ptf/bolometric.ptf',
            pbset='Bolometric',
            pbname='900-40000',
            wlunits=u.m,
            calibrated=True,
            reference='Flat response to simulate bolometric throughput',
            version=2.5
        )

    pb.version = 2.5
    pb.add_to_history('TMAP model atmospheres added.')
    pb.content = []

    pb.compute_blackbody_intensities(include_extinction=False)

    for atm in ['ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO']:
        pb.compute_intensities(atm=atm, path=f'tables/{atm}', verbose=True)
        pb.compute_ldcoeffs(ldatm=atm)
        pb.compute_ldints(ldatm=atm)

    pb.save('bolometric.fits')

    try:
        pb = Passband.load('tables/passbands/johnson_v.fits')
    except FileNotFoundError:
        pb = Passband(
            ptf='tables/ptf/johnson_v.ptf',
            pbset='Johnson',
            pbname='V',
            wlunits=u.AA,
            calibrated=True,
            reference='Maiz Apellaniz (2006), AJ 131, 1184',
            version=2.5,
            comment=''
        )

    pb.version = 2.5
    pb.add_to_history('TMAP model atmospheres added.')
    pb.content = []

    pb.compute_blackbody_intensities(include_extinction=True)

    for atm in ['ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO']:
        pb.compute_intensities(atm=atm, path=f'tables/{atm}', verbose=True)
        pb.compute_ldcoeffs(ldatm=atm)
        pb.compute_ldints(ldatm=atm)

    pb.import_wd_atmcof('tables/wd/atmcofplanck.dat', 'tables/wd/atmcof.dat', 7)

    pb.save('johnson_v.fits')
