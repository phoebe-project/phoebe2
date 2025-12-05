from phoebe import __version__ as phoebe_version
from phoebe import conf, mpi
from phoebe.utils import _bytes
from phoebe.atmospheres import models
from tqdm import tqdm

import ndpolator

# NOTE: we'll import directly from astropy here to avoid
# circular imports BUT any changes to these units/constants
# inside phoebe will be ignored within passbands
from astropy.constants import h, c, k_B
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

import numpy as np
from scipy import interpolate, integrate
from scipy.optimize import least_squares
from packaging.version import parse
from datetime import datetime
import libphoebe
import os
import sys
import shutil
import json
import time

# NOTE: python3 only
from urllib.request import urlopen, urlretrieve

from phoebe.utils import parse_json

import logging
logger = logging.getLogger("PASSBANDS")
logger.addHandler(logging.NullHandler())

# define the URL to query for online passbands.  See tables.phoebe-project.org
# repo for the source-code of the server
_url_tables_server = os.getenv('PHOEBE_TABLES_SERVER', 'https://tables.phoebe-project.org')
# comment out the following line if testing tables.phoebe-project.org server locally:
# _url_tables_server = 'http://localhost:5555'

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


class InterpQuery:
    def __init__(self, cols, pts, meta=None):
        """
        A simple class to hold query information for model atmosphere
        interpolation.

        Arguments
        ---------
        * `cols` (list of strings): names of the columns in the query table
        * `pts` (2D array): points in the query table; shape is (N, len(cols))
        * `meta` (dict, optional, default=None): any additional metadata
            associated with the query

        Returns
        -------
        * an instatiated <phoebe.atmospheres.passbands.InterpQuery> object.
        """

        self.cols = list(cols)
        self.pts = np.ascontiguousarray(pts)
        if self.pts.ndim != 2 or self.pts.shape[1] != len(self.cols):
            raise ValueError(f"Shape mismatch: pts.shape={self.pts.shape}, len(cols)={len(self.cols)}")
        self.meta = meta if meta is not None else {}

    def index(self, name):
        try:
            return self.cols.index(name)
        except ValueError:
            raise KeyError(f"Column '{name}' not found")

    def get_column(self, name):
        idx = self.index(name)
        return self.pts[:, idx]

    def subset(self, names):
        idxs = [self.index(name) for name in names]
        return InterpQuery([self.cols[i] for i in idxs], np.ascontiguousarray(self.pts[:, idxs]), meta=self.meta)

    def __getitem__(self, key):
        subset_pts = self.pts[key]

        if subset_pts.ndim == 1:
            subset_pts = subset_pts.reshape(1, -1)
        return InterpQuery(self.cols, np.ascontiguousarray(subset_pts), meta=self.meta)

    def __len__(self):
        """Return number of query points"""
        return self.pts.shape[0]

    def __repr__(self):
        return f'<InterpQuery: {len(self)} points, cols={self.cols}, meta_keys={list(self.meta.keys())}>'


class InterpResult:
    def __init__(self, interps, **kwargs):
        """
        A class to hold interpolation results from ndpolator operations.

        Arguments
        ---------
        * `interps` (array, required): interpolated values
        * `**kwargs`: any additional results or metadata
        """
        self.interps = interps
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_ndpolator(cls, ndp_output):
        """
        Create an InterpResult from an ndpolator dictionary result.
        
        Arguments
        ---------
        * `ndp_output` (dict): dictionary returned by ndpolator.ndpolate()

        Returns
        -------
        * InterpResult instance
        """
        if not isinstance(ndp_output, dict):
            raise TypeError("ndp_output must be a dictionary")

        if 'interps' not in ndp_output:
            raise ValueError("ndp_output must contain 'interps' key")

        return cls(**ndp_output)

    @property
    def shape(self):
        """Shape of interpolated results"""
        return self.interps.shape

    @property
    def size(self):
        """Size of interpolated results"""
        return self.interps.size

    def get_interpolated_values(self):
        return getattr(self, 'interps', None)

    def get_distances(self):
        return getattr(self, 'dists', None)

    def get_bfs(self):
        return getattr(self, 'bfs', None)

    def __len__(self):
        """Number of interpolated points"""
        return len(self.interps)

    def __getitem__(self, key):
        """Allow slicing of results"""
        sliced_interps = self.interps[key]
        
        # Handle single row slicing to maintain 2D structure when appropriate
        if isinstance(key, int) and np.ndim(sliced_interps) == 1 and np.ndim(self.interps) == 2:
            # Reshape single row to maintain 2D structure
            sliced_interps = sliced_interps.reshape(1, -1)
        
        sliced_attrs = {'interps': sliced_interps}
        
        # Handle other array-like attributes that should be sliced along the same dimension
        for attr_name, attr_value in self.__dict__.items():
            if attr_name == 'interps':
                continue
            elif attr_value is None:
                sliced_attrs[attr_name] = None
            elif hasattr(attr_value, '__getitem__') and hasattr(attr_value, 'shape'):
                # This is an array-like object, slice it appropriately
                try:
                    if isinstance(key, tuple) and len(key) == 2:
                        # For 2D slicing like s[ld_func] = np.s_[:, 7:11]
                        sliced_value = attr_value[key[0]]  # Only slice rows
                    else:
                        sliced_value = attr_value[key]
                    
                    # Apply same reshaping logic for single row slices
                    if isinstance(key, int) and np.ndim(sliced_value) == 1 and np.ndim(attr_value) == 2:
                        sliced_value = sliced_value.reshape(1, -1)
                    
                    sliced_attrs[attr_name] = sliced_value
                except (IndexError, TypeError):
                    # If slicing fails, keep the original
                    sliced_attrs[attr_name] = attr_value
            else:
                # Keep scalar/non-array attributes unchanged
                sliced_attrs[attr_name] = attr_value
        
        return InterpResult(**sliced_attrs)

    def __repr__(self):
        return f'<InterpResult: {self.shape}>'


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

        Step #3: instantiate a model atmosphere object and compute intensities:

        ```py atm = CK2004ModelAtmosphere('path/to/ck2004')
        pb.compute_intensities(atm) ```

        Step #4: repeat step #3 for other model atmospheres.

        Step #5: -- optional -- import WD tables for comparison. This can only
        be done if the passband is in the list of supported passbands in WD.
        The WD index of the passband is passed to the import_wd_atmcof()
        function below as the last argument.

        ```py from phoebe.atmospheres import atmcof atmdir =
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
        'tables/wd')) atmcof.init(atmdir+'/atmcofplanck.dat',
        atmdir+'/atmcof.dat') pb.import_wd_atmcof(atmdir+'/atmcofplanck.dat',
        atmdir+'/atmcof.dat', 7) ```

        Step #6: save the passband file:

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

        self.add_to_history('passband transmission function updated.')

    def save(self, archive, overwrite=True, update_timestamp=True, export_to_pre25=False):
        """
        Saves the passband file in the fits format.

        Arguments
        ----------
        * `archive` (string): filename of the saved file
        * `overwrite` (bool, optional, default=True): whether to overwrite an
            existing file with the same filename as provided in `archive`
        * `update_timestamp` (bool, optional, default=True): whether to update
            the stored timestamp with the current time.
        * `export_to_pre25` (bool, optional, default=False): whether to export
            the passband file to a pre-2.5 format. This includes renaming the
            columns in the tables to match the old passband files, exporting
            Inorm tables for model atmospheres, exporting blackbody functions
            and exporting legacy comments.
        """

        # Timestamp is used for passband versioning.
        timestamp = time.ctime() if update_timestamp else self.timestamp

        header = fits.Header()
        if export_to_pre25:
            header['PHOEBEVN'] = '2.4.17'
        else:
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

        if export_to_pre25:
            header['COMMENTS'] = ''

        # We build content from scratch to avoid any potential issues with unsupported tables:
        content = []

        # Add history entries:
        if export_to_pre25:
            header['history'] = '-END-'.join(self.history) + '-END-'
        else:
            for entry in self.history:
                header['history'] = entry

        # Add comments:
        for comment in self.comments:
            header['comment'] = comment

        if 'extern_planckint:Inorm' in self.content or 'extern_atmx:Inorm' in self.content:
            header['WD_IDX'] = self.extern_wd_idx
            content.append('extern_planckint:Inorm')
            content.append('extern_atmx:Inorm')

        data = []

        primary_hdu = fits.PrimaryHDU(header=header)
        data.append(primary_hdu)

        data.append(fits.table_to_hdu(Table(self.ptf_table, meta={'extname': 'PTFTABLE'})))

        # axes:
        for atm in models._atmtable:
            if atm.external:
                continue

            if f'{atm.name}:Inorm' in self.content and f'{atm.name}:Imu' not in self.content:
                basic_axes = self.ndp[atm.name].axes

                for name, axis in zip(atm.basic_axis_names, basic_axes):
                    if export_to_pre25:
                        data.append(fits.table_to_hdu(Table({name[:-1]: axis}, meta={'extname': f'{atm.prefix}_{name}'})))
                    else:
                        data.append(fits.table_to_hdu(Table({name: axis}, meta={'extname': f'{atm.prefix}_{name}'})))

            if f'{atm.name}:Imu' in self.content:
                basic_axes = self.ndp[atm.name].axes
                associated_axes = self.ndp[atm.name].table['imu@photon']['associated_axes']

                for name, axis in zip(atm.basic_axis_names + ['mus'], basic_axes + associated_axes):
                    if export_to_pre25:
                        data.append(fits.table_to_hdu(Table({name[:-1]: axis}, meta={'extname': f'{atm.prefix}_{name}'})))
                    else:
                        data.append(fits.table_to_hdu(Table({name: axis}, meta={'extname': f'{atm.prefix}_{name}'})))

            if f'{atm.name}:ext' in self.content:
                associated_axes = self.ndp[atm.name].table['ext@photon']['associated_axes']

                for name, axis in zip(['ebvs', 'rvs'], associated_axes):
                    if export_to_pre25:
                        data.append(fits.table_to_hdu(Table({name[:-1]: axis}, meta={'extname': f'{atm.prefix}_{name}'})))
                    else:
                        data.append(fits.table_to_hdu(Table({name: axis}, meta={'extname': f'{atm.prefix}_{name}'})))

        # grids:
        for atm in models._atmtable:
            if atm.external:
                continue

            if f'{atm.name}:Inorm' in self.content:
                if export_to_pre25 and atm.name == 'blackbody':
                    teffs = self.ndp['blackbody'].axes[0]
                    log10ints = self.ndp['blackbody'].table['inorm@energy']['grid']
                    bb_func_energy = interpolate.splrep(teffs, log10ints, s=0)
                    log10ints = self.ndp['blackbody'].table['inorm@photon']['grid']
                    bb_func_photon = interpolate.splrep(teffs, log10ints, s=0)

                    bb_func = Table({'teff': bb_func_energy[0], 'logi_e': bb_func_energy[1], 'logi_p': bb_func_photon[1]}, meta={'extname': 'BB_FUNC'})
                    data.append(fits.table_to_hdu(bb_func))
                else:
                    data.append(fits.ImageHDU(self.ndp[atm.name].table['inorm@energy']['grid'], name=f'{atm.prefix}negrid'))
                    data.append(fits.ImageHDU(self.ndp[atm.name].table['inorm@photon']['grid'], name=f'{atm.prefix}npgrid'))

                content.append(f'{atm.name}:Inorm')

            if f'{atm.name}:Imu' in self.content:
                data.append(fits.ImageHDU(self.ndp[atm.name].table['imu@energy']['grid'], name=f'{atm.prefix.upper()}FEGRID'))
                data.append(fits.ImageHDU(self.ndp[atm.name].table['imu@photon']['grid'], name=f'{atm.prefix.upper()}FPGRID'))
                content.append(f'{atm.name}:Imu')

                if export_to_pre25 and f'{atm.name}:Inorm' not in content:
                    data.append(fits.ImageHDU(self.ndp[atm.name].table['imu@energy']['grid'][..., -1, :], name=f'{atm.prefix.upper()}NEGRID'))
                    data.append(fits.ImageHDU(self.ndp[atm.name].table['imu@photon']['grid'][..., -1, :], name=f'{atm.prefix.upper()}NPGRID'))
                    content.append(f'{atm.name}:Inorm')

            if f'{atm.name}:ld' in self.content:
                data.append(fits.ImageHDU(self.ndp[atm.name].table['ld@energy']['grid'], name=f'{atm.prefix.upper()}LEGRID'))
                data.append(fits.ImageHDU(self.ndp[atm.name].table['ld@photon']['grid'], name=f'{atm.prefix.upper()}LPGRID'))
                content.append(f'{atm.name}:ld')

            if f'{atm.name}:ldint' in self.content:
                data.append(fits.ImageHDU(self.ndp[atm.name].table['ldint@energy']['grid'], name=f'{atm.prefix.upper()}IEGRID'))
                data.append(fits.ImageHDU(self.ndp[atm.name].table['ldint@photon']['grid'], name=f'{atm.prefix.upper()}IPGRID'))
                content.append(f'{atm.name}:ldint')

            if f'{atm.name}:ext' in self.content:
                if export_to_pre25 and atm.name == 'blackbody':
                    data.append(fits.ImageHDU(self.ndp[atm.name].table['ext@energy']['grid'], name=f'{atm.prefix.upper()}EGRID'))
                    data.append(fits.ImageHDU(self.ndp[atm.name].table['ext@photon']['grid'], name=f'{atm.prefix.upper()}PGRID'))
                else:
                    data.append(fits.ImageHDU(self.ndp[atm.name].table['ext@energy']['grid'], name=f'{atm.prefix.upper()}XEGRID'))
                    data.append(fits.ImageHDU(self.ndp[atm.name].table['ext@photon']['grid'], name=f'{atm.prefix.upper()}XPGRID'))
                content.append(f'{atm.name}:ext')

        # All saved content has been syndicated to the content list:
        primary_hdu.header['CONTENT'] = str(content)

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

        # NOTE: all passband files on the server are stored in the new
        # format, so it's not necessary to test for legacy constructs,
        # in particular for 'comments' and 'history' entries.

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
                stored_atms = set([content.split(':')[0] for content in self.content])

                # TODO: replace with < parse('2.5.0') when 2.5.0 is released
                if parse(self.phoebe_version) != parse('2.4.17.dev+feature-blending'):
                    if 'blackbody' in stored_atms:
                        # blackbody atmospheres are reworked, so we need to
                        # recompute the intensities:
                        self.compute_intensities(
                            atm=models.BlackbodyModelAtmosphere(),
                            include_mus=False,
                            include_ld=False,
                            include_extinction='blackbody:ext' in self.content,
                            add_history_entry=False,
                            verbose=False
                        )
                        ndp = self.ndp['blackbody']  # initialized by compute_intensities()

                        # store axes:
                        hdul['bb_teffs'] = fits.table_to_hdu(Table({'teffs': ndp.axes[0]}, meta={'extname': 'bb_teffs'}))
                        if 'blackbody:ext' in self.content:
                            associated_axes = ndp.table['ext@photon']['associated_axes']
                            hdul['bb_ebvs'] = fits.table_to_hdu(Table({'ebvs': associated_axes[0]}, meta={'extname': 'bb_ebvs'}))
                            hdul['bb_rvs'] = fits.table_to_hdu(Table({'rvs': associated_axes[1]}, meta={'extname': 'bb_rvs'}))

                        # store tables:
                        hdul.append(fits.ImageHDU(self.ndp['blackbody'].table['inorm@energy']['grid'], name='bbnegrid'))
                        hdul.append(fits.ImageHDU(self.ndp['blackbody'].table['inorm@photon']['grid'], name='bbnpgrid'))
                        if 'blackbody:ext' in self.content:
                            hdul.append(fits.ImageHDU(self.ndp['blackbody'].table['ext@energy']['grid'], name='bbxegrid'))
                            hdul.append(fits.ImageHDU(self.ndp['blackbody'].table['ext@photon']['grid'], name='bbxpgrid'))

                        # clean up:
                        del self.ndp['blackbody']

                    # rename columns in old tables:
                    if 'ck2004' in stored_atms:
                        hdul['ck_teffs'].data.columns.change_name('teff', 'teffs')
                        hdul['ck_loggs'].data.columns.change_name('logg', 'loggs')
                        hdul['ck_abuns'].data.columns.change_name('abun', 'abuns')

                        if 'ck2004:ext' in self.content:
                            hdul['ck_ebvs'].data.columns.change_name('ebv', 'ebvs')
                            hdul['ck_rvs'].data.columns.change_name('rv', 'rvs')

                    if 'phoenix' in stored_atms:
                        hdul['ph_teffs'].data.columns.change_name('teff', 'teffs')
                        hdul['ph_loggs'].data.columns.change_name('logg', 'loggs')
                        hdul['ph_abuns'].data.columns.change_name('abun', 'abuns')

                        if 'phoenix:ext' in self.content:
                            hdul['ph_ebvs'].data.columns.change_name('ebv', 'ebvs')
                            hdul['ph_rvs'].data.columns.change_name('rv', 'rvs')

                if 'extern_planckint:Inorm' in self.content or 'extern_atmx:Inorm' in self.content:
                    atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/wd'))
                    planck = os.path.join(atmdir+'/atmcofplanck.dat').encode('utf8')
                    atm_file = os.path.join(atmdir+'/atmcof.dat').encode('utf8')

                    self.wd_data = libphoebe.wd_readdata(planck, atm_file)
                    self.extern_wd_idx = header['wd_idx']

                # Model atmospheres in the passband file:
                stored_atms = set([entry.split(':')[0] for entry in self.content])

                # We have to iterate over available atms rather than stored atms because
                # the stored atms may not be available in the current version of PHOEBE.
                available_atms = models.ModelAtmosphere.__subclasses__()
                for atm in available_atms:
                    if atm.name not in stored_atms:
                        continue

                    if atm.external:
                        continue

                    basic_axes = tuple([np.array(list(hdul[f'{atm.prefix}_{name}'].data[name])) for name in atm.basic_axis_names])
                    # basic_axes = tuple(np.asarray(hdul[f'{atm.prefix}_{name}'].data[name]) for name in atm.basic_axis_names)
                    self.ndp[atm.name] = ndpolator.Ndpolator(basic_axes=basic_axes)

                    if f'{atm.name}:Inorm' in self.content:
                        self.ndp[atm.name].register(name='inorm@photon', associated_axes=None, grid=hdul[f'{atm.prefix}npgrid'].data)
                        self.ndp[atm.name].register(name='inorm@energy', associated_axes=None, grid=hdul[f'{atm.prefix}negrid'].data)

                    if f'{atm.name}:Imu' in self.content:
                        atm_photon_grid = hdul[f'{atm.prefix}fpgrid'].data
                        atm_energy_grid = hdul[f'{atm.prefix}fegrid'].data

                        # normal passband intensities:
                        self.ndp[atm.name].register(name='inorm@photon', associated_axes=None, grid=atm_photon_grid[..., -1, :])
                        self.ndp[atm.name].register(name='inorm@energy', associated_axes=None, grid=atm_energy_grid[..., -1, :])

                        # specific passband intensities:
                        self.ndp[atm.name].register(name='imu@photon', associated_axes=(atm.mus,), grid=atm_photon_grid)
                        self.ndp[atm.name].register(name='imu@energy', associated_axes=(atm.mus,), grid=atm_energy_grid)

                    if f'{atm.name}:ld' in self.content:
                        self.ndp[atm.name].register(name='ld@photon', associated_axes=None, grid=hdul[f'{atm.prefix}legrid'].data)
                        self.ndp[atm.name].register(name='ld@energy', associated_axes=None, grid=hdul[f'{atm.prefix}lpgrid'].data)

                    if f'{atm.name}:ldint' in self.content:
                        self.ndp[atm.name].register(name='ldint@photon', associated_axes=None, grid=hdul[f'{atm.prefix}iegrid'].data)
                        self.ndp[atm.name].register(name='ldint@energy', associated_axes=None, grid=hdul[f'{atm.prefix}ipgrid'].data)

                    if f'{atm.name}:ext' in self.content:
                        # associated axes:
                        ebvs = np.array(list(hdul[f'{atm.prefix}_ebvs'].data['ebvs']))
                        rvs = np.array(list(hdul[f'{atm.prefix}_rvs'].data['rvs']))

                        self.ndp[atm.name].register(name='ext@photon', associated_axes=(ebvs, rvs), grid=hdul[f'{atm.prefix}xegrid'].data)
                        self.ndp[atm.name].register(name='ext@energy', associated_axes=(ebvs, rvs), grid=hdul[f'{atm.prefix}xpgrid'].data)

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

    def ld_func(self, mu=1.0, ld_coeffs=np.array([[0.5]]), ld_func='linear'):
        ld_coeffs = np.atleast_2d(ld_coeffs)

        if ld_func == 'linear':
            return 1-ld_coeffs[:, 0]*(1-mu)
        elif ld_func == 'logarithmic':
            return 1-ld_coeffs[:, 0]*(1-mu)-ld_coeffs[:, 1]*mu*np.log(np.maximum(mu, 1e-6))
        elif ld_func == 'square_root':
            return 1-ld_coeffs[:, 0]*(1-mu)-ld_coeffs[:, 1]*(1-np.sqrt(mu))
        elif ld_func == 'quadratic':
            return 1-ld_coeffs[:, 0]*(1-mu)-ld_coeffs[:, 1]*(1-mu)*(1-mu)
        elif ld_func == 'power':
            return 1-ld_coeffs[:, 0]*(1-np.sqrt(mu))-ld_coeffs[:, 1]*(1-mu)-ld_coeffs[:, 2]*(1-mu*np.sqrt(mu))-ld_coeffs[:, 3]*(1.0-mu*mu)
        else:
            raise NotImplementedError(f'ld_func={ld_func} is not supported.')

    def compute_intensities(self, atm, include_mus=True, include_ld=True, include_extinction=False, rvs=None, ebvs=None, add_history_entry=True, verbose=True):
        """
        Computes direction-dependent passband intensities using the passed `atm`
        model atmospheres.

        Arguments
        ----------
        * `atm` (<models.ModelAtmosphere> subclass): model atmosphere to use for the
            computation.
        * `include_mus` (bool, optional, default=True): set to True to include
            specific angles in the computation.
        * `include_ld` (bool, optional, default=True): set to True to include
            limb darkening coefficients in the computation. This will also
            calculate and tabulate integrals of the piecewise linear limb
            darkening function.
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
        * `add_history_entry` (bool, optional, default=True): set to True to add
            a history entry to the passband file.
        * `verbose` (bool, optional, default=True): set to True to display
            progress in the terminal.

        Raises
        ------
        * ValueError: if the `atm` instance does not have the wavelength span
            defined.
        """

        if verbose:
            print(f"Computing {atm.name} specific passband intensities for {self.pbset}:{self.pbname} {'with' if include_extinction else 'without'} extinction.")

        # Model atmosphere either needs to tabulate intensities as a
        # function of wavelength or provide a function to compute them.
        can_compute_intensity = hasattr(atm, 'intensity') and callable(atm.intensity)

        if not hasattr(atm, 'wls') and not can_compute_intensity:
            raise ValueError(f'Model atmosphere {atm.name} does not have wavelength span defined.')

        # Same for specific angles:
        if include_mus and not hasattr(atm, 'mus') and not can_compute_intensity:
            raise ValueError(f'Model atmosphere {atm.name} does not have specific angles defined.')

        # Preliminary checks passed, we can instantiate an ndpolator instance:
        self.ndp[atm.name] = ndpolator.Ndpolator(basic_axes=atm.basic_axes)

        if can_compute_intensity:
            # If the model atmosphere defines an intensity function, use the passband itself:
            wls = np.array(self.ptf_table['wl'], dtype=np.float64)
            ptf = np.array(self.ptf_table['fl'], dtype=np.float64)
        else:
            # trim wavelengths to the passband limits:
            keep = (atm.wls >= self.ptf_table['wl'][0]) & (atm.wls <= self.ptf_table['wl'][-1])
            wls = atm.wls[keep]
            ptf = self.ptf(wls)

        if include_mus:
            grid_shape = tuple([len(axis) for axis in atm.basic_axes + (atm.mus,)] + [1])
        else:
            grid_shape = tuple([len(axis) for axis in atm.basic_axes] + [1])

        # initialize intensity arrays:
        atm_energy_grid = np.full(shape=grid_shape, fill_value=np.nan)
        atm_photon_grid = np.full_like(atm_energy_grid, fill_value=np.nan)

        if include_extinction:
            # add extinction axes:
            if ebvs is None:
                ebvs = np.linspace(0., 3., 30)
            if rvs is None:
                rvs = np.linspace(2., 6., 16)

            # initialize arrays for extincted intensities:
            associated_axes = (ebvs, rvs)
            grid_shape = tuple([len(axis) for axis in atm.basic_axes + associated_axes] + [1])
            ext_photon_grid = np.empty(shape=grid_shape)
            ext_energy_grid = np.empty_like(ext_photon_grid)

            axbx = libphoebe.gordon_extinction(wls)
            ax, bx = axbx[:,0], axbx[:,1]

            # The following code broadcasts arrays so that integration can be vectorized:
            Alam = 10**(-0.4 * ebvs[None, :, None] * (rvs[None, None, :] * ax[:, None, None] + bx[:, None, None]))

        if can_compute_intensity:
            ints = atm.intensity(wls)  # must be in in W/m^3
            pbints_energy = ptf*ints
            fluxes_energy = np.trapz(pbints_energy, wls)
            fluxes_energy = atm.limb_treatment(fluxes_energy)

            pbints_photon = wls * pbints_energy
            fluxes_photon = np.trapz(pbints_photon, wls)
            fluxes_photon = atm.limb_treatment(fluxes_photon)

            atm_energy_grid = np.log10(fluxes_energy/self.ptf_area).reshape(-1, 1)
            atm_photon_grid = np.log10(fluxes_photon/self.ptf_photon_area).reshape(-1, 1)

            self.ndp[atm.name].register('inorm@photon', None, atm_photon_grid)
            self.ndp[atm.name].register('inorm@energy', None, atm_energy_grid)

            if include_extinction:
                egrid = np.trapz(pbints_energy[:, :, None, None] * Alam[None, :, :, :], x=wls, axis=1) / np.trapz(pbints_energy[:, :, None, None], x=wls, axis=1)
                pgrid = np.trapz(pbints_photon[:, :, None, None] * Alam[None, :, :, :], x=wls, axis=1) / np.trapz(pbints_photon[:, :, None, None], x=wls, axis=1)

                ext_energy_grid = egrid.reshape(len(atm.teffs), len(ebvs), len(rvs), 1)
                ext_photon_grid = pgrid.reshape(len(atm.teffs), len(ebvs), len(rvs), 1)
                self.ndp[atm.name].register('ext@energy', (ebvs, rvs), ext_energy_grid)
                self.ndp[atm.name].register('ext@photon', (ebvs, rvs), ext_photon_grid)
                if f'{atm.name}:ext' not in self.content:
                    self.content.append(f'{atm.name}:ext')

            if f'{atm.name}:Inorm' not in self.content:
                self.content.append(f'{atm.name}:Inorm')

            if add_history_entry:
                self.add_to_history(f"{atm.name} intensities {'with' if include_extinction else 'w/o'} extinction added.")

            return

        for i, model in tqdm(enumerate(atm.models), desc=atm.name, total=atm.nmodels, disable=not verbose, unit=' models'):
            with fits.open(model) as hdu:
                # load specific intensities and trim them to the passband limits:
                ints = hdu[0].data[:, keep] * atm.units  # must be in in W/m^3

            # calculate energy-weighted passband intensities and fluxes:
            pbints_energy = ptf*ints
            fluxes_energy = np.trapz(pbints_energy, wls)

            # calculate photon count-weighted passband intensities and fluxes:
            pbints_photon = wls*pbints_energy
            fluxes_photon = np.trapz(pbints_photon, wls)

            # handle the limb according to the prescription in the model atmosphere:
            fluxes_energy = atm.limb_treatment(fluxes_energy)
            fluxes_photon = atm.limb_treatment(fluxes_photon)

            # compute specific energy-weighted and photon count-weighted intensities:
            atm_energy_grid[tuple(atm.indices[i])] = np.log10(fluxes_energy/self.ptf_area).reshape(-1, 1)
            atm_photon_grid[tuple(atm.indices[i])] = np.log10(fluxes_photon/self.ptf_photon_area).reshape(-1, 1)

            if include_extinction:
                # we only use normal emergent intensities for extinction:
                epbints = pbints_energy[-1].reshape(-1, 1)
                egrid = np.trapz(epbints[:, :, None, None] * Alam[:, None, :, :], wls, axis=0) / np.trapz(epbints[:, :, None, None], wls, axis=0)

                ppbints = pbints_photon[-1].reshape(-1, 1)
                pgrid = np.trapz(ppbints[:, :, None, None] * Alam[:, None, :, :], wls, axis=0) / np.trapz(ppbints[:, :, None, None], wls, axis=0)

                ext_energy_grid[tuple(atm.indices[i])] = egrid.reshape(len(ebvs), len(rvs), 1)
                ext_photon_grid[tuple(atm.indices[i])] = pgrid.reshape(len(ebvs), len(rvs), 1)

        self.ndp[atm.name].register('inorm@photon', None, atm_photon_grid[..., -1, :])
        self.ndp[atm.name].register('inorm@energy', None, atm_energy_grid[..., -1, :])
        if f'{atm.name}:Inorm' not in self.content:
            self.content.append(f'{atm.name}:Inorm')

        self.ndp[atm.name].register('imu@photon', (atm.mus,), atm_photon_grid)
        self.ndp[atm.name].register('imu@energy', (atm.mus,), atm_energy_grid)
        if f'{atm.name}:Imu' not in self.content:
            self.content.append(f'{atm.name}:Imu')

        if include_extinction:
            self.ndp[atm.name].register('ext@photon', (ebvs, rvs), ext_photon_grid)
            self.ndp[atm.name].register('ext@energy', (ebvs, rvs), ext_energy_grid)
            if f'{atm.name}:ext' not in self.content:
                self.content.append(f'{atm.name}:ext')

        if add_history_entry:
            self.add_to_history(f"{atm.name} intensities {'with' if include_extinction else 'w/o'} extinction added.")

        if include_ld:
            if verbose:
                print(f'Computing {atm.name} limb darkening coefficients...')

            # initialize arrays for limb darkening coefficients:
            ld_energy_grid = np.full(shape=[len(axis) for axis in atm.basic_axes]+[11], fill_value=np.nan)
            ld_photon_grid = np.full_like(ld_energy_grid, fill_value=np.nan)

            # initialize arrays for limb darkening integrals:
            ldint_energy_grid = np.full(shape=[len(axis) for axis in atm.basic_axes]+[1], fill_value=np.nan)
            ldint_photon_grid = np.full_like(ldint_energy_grid, fill_value=np.nan)

            # define the residuals function for the least squares optimization:
            def ld_resids(x, *args, **kwargs):
                xdata = kwargs.get('xdata')
                ydata = kwargs.get('ydata')
                ld_func = kwargs.get('ld_func', 'linear')

                return self.ld_func(mu=xdata, ld_coeffs=x, ld_func=ld_func) - ydata

            # loop over all defined coordinates to compute limb darkening coefficients and integrals:
            for grid, ld_grid, ldint_grid in zip([atm_energy_grid, atm_photon_grid], [ld_energy_grid, ld_photon_grid], [ldint_energy_grid, ldint_photon_grid]):
                for ind in atm.indices:
                    xdata = atm.mus
                    ydata = 10**grid[tuple(ind)].flatten()
                    ydata /= ydata[-1]
                    # TODO: consider support for non-uniform weighing

                    ld_row = []
                    for ld_func, ld_dim in zip(['linear', 'logarithmic', 'square_root', 'quadratic', 'power'], [1, 2, 2, 2, 4]):
                        result = least_squares(fun=ld_resids, x0=np.full(ld_dim, 0.5), method='lm', kwargs={'xdata': xdata, 'ydata': ydata, 'ld_func': ld_func})
                        ld_row.append(result.x)

                    ld_grid[tuple(ind)] = np.hstack(ld_row)

                    # compute limb darkening integrals for the piecewise linear LD func:
                    slopes = np.diff(ydata)/np.diff(xdata)
                    intercepts = ydata[:-1] - slopes*xdata[:-1]
                    areas = 2/3 * slopes * np.diff(xdata**3) + intercepts * np.diff(xdata**2)
                    ldint_grid[tuple(ind)] = areas.sum()

            self.ndp[atm.name].register('ld@photon', None, ld_photon_grid)
            self.ndp[atm.name].register('ld@energy', None, ld_energy_grid)
            if f'{atm.name}:ld' not in self.content:
                self.content.append(f'{atm.name}:ld')
            
            if add_history_entry:
                self.add_to_history(f'LD coefficients for {atm.name} added.')

            self.ndp[atm.name].register('ldint@photon', None, ldint_photon_grid)
            self.ndp[atm.name].register('ldint@energy', None, ldint_energy_grid)
            if f'{atm.name}:ldint' not in self.content:
                self.content.append(f'{atm.name}:ldint')

            if add_history_entry:
                self.add_to_history(f'LD integrals for {atm.name} added.')

    def interpolate_ldcoeffs(self, query, ldatm=models.CK2004ModelAtmosphere, ld_func='power', intens_weighting='photon', ld_extrapolation_method='none'):
        """
        Interpolate the passband-stored table of LD model coefficients.

        Arguments
        ------------
        * `query` (<InterpQuery>, required): the interpolation query object.
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
            'linear': np.s_[:, :1],
            'logarithmic': np.s_[:, 1:3],
            'square_root': np.s_[:, 3:5],
            'quadratic': np.s_[:, 5:7],
            'power': np.s_[:, 7:11],
            'all': np.s_[:, :]
        }

        if ld_func not in s.keys():
            raise ValueError(f'ld_func={ld_func} is invalid; valid options are {s.keys()}.')

        if f'{ldatm.name}:ld' not in self.content:
            raise ValueError(f'limb darkening coefficients for ldatm={ldatm.name} not found for the {self.pbset}:{self.pbname} passband.')

        # limb darkening coefficients depend only on basic axes:
        ndp_output = self.ndp[ldatm.name].ndpolate(
            f'ld@{intens_weighting}',
            query_pts=query.subset(ldatm.basic_axis_names).pts,
            extrapolation_method=ld_extrapolation_method
        )

        return InterpResult.from_ndpolator(ndp_output)[s[ld_func]]

    def interpolate_extinct(self, query, atm=models.CK2004ModelAtmosphere, intens_weighting='photon', extrapolation_method='none'):
        """
        Interpolates the passband-stored tables of extinction corrections

        Arguments
        ----------
        * `query`
        * `atm`
        * `intens_weighting`
        * `extrapolation_method`

        Returns
        ---------
        * extinction factor

        Raises
        --------
        * ValueError if `atm` is not supported.
        """

        if f'{atm.name}:ext' not in self.content:
            raise ValueError(f'extinction factors for atm={atm.name} not found for the {self.pbset}:{self.pbname} passband.')

        # extinction coefficients depend on basic axes, ebvs and rvs:
        ndp_output = self.ndp[atm.name].ndpolate(
            f'ext@{intens_weighting}',
            query_pts=query.subset(atm.basic_axis_names + ['ebvs', 'rvs']).pts,
            extrapolation_method=extrapolation_method
        )

        return InterpResult.from_ndpolator(ndp_output)

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
            raise ValueError(f'wdidx value out of bounds: 1 <= wdidx <= {Npb}')

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

        self.add_to_history('Wilson-Devinney atmosphere tables imported.')

    def interpolate_inorms(self, query, atm=models.CK2004ModelAtmosphere, ldatm=models.CK2004ModelAtmosphere, ldint=None, ld_func='interp', ld_coeffs=None, intens_weighting='photon', atm_extrapolation_method='none', ld_extrapolation_method='none', blending_method='none', blending_margin=3, dist_threshold=1e-5):
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
        | tremblay  |               |                         |           |                  |                                                             |

        Arguments
        ----------
        * `query` (<InterpQuery>, required): the interpolation query object.
        * `atm` (<models.ModelAtmosphere>, optional,
          default=CK2004ModelAtmosphere): model atmosphere to be used for
          calculation
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
          weighting switch
        * `atm_extrapolation_method` (string, optional, default='none'): the
          method for off-grid intensity extrapolation ('none', 'nearest',
          'linear'). Option 'none' will return a nan for off-grid points;
          'nearest' will use the nearest grid point value; 'linear' will use
          linear extrapolation.
        * `ld_extrapolation_method` (string, optional, default='none'): the
          method for off-grid limb darkening extrapolation ('none', 'nearest', 
          'linear'). See `atm_extrapolation_method` for details on options.
        * `blending_method` (string, optional, default='none'): the method to
          blend model atmosphere with blackbody ('none' or 'blackbody'). Option
          'none' will not do blending; 'blackbody' will use blackbody
          intensities for off-grid points and blend the model into the
          blackbody over a distance defined by `blending_margin`.
        * `dist_threshold` (float, optional, default=1e-5): off-grid distance
          threshold. Query points farther than this value, in hypercube-
          normalized units, are considered off-grid.
        * `blending_margin` (float, optional, default=3): the off-grid region,
          in hypercube-normalized units, where blending should be done.

        Returns
        ----------
        * (dict) a dict of normal emergent passband intensities and associated
          values. Dictionary keys are: 'inorms' (normal intensities),
          and 'bfs' (blending factors, only if blending_method is not
          'none').

        Raises
        ----------
        * ValueError: if atmosphere parameters are out of bounds for the
          table.
        * NotImplementedError: if `ld_func` is not supported.
        """

        if f'{atm.name}:Inorm' not in self.content:
            raise ValueError(f'atm={atm.name} tables are not available in the {self.pbset}:{self.pbname} passband.')

        if atm.name == 'blackbody':
            # compute normal emergent blackbody intensities:
            log_inorms = InterpResult.from_ndpolator(
                self.ndp[atm.name].ndpolate(
                    f'inorm@{intens_weighting}',
                    query_pts=query.subset(atm.basic_axis_names).pts,
                    extrapolation_method=atm_extrapolation_method
                )
            ).get_interpolated_values()

            # correct normal intensities for integrated limb darkening:
            if ldint is None:
                ldint = self.interpolate_ldints(
                    query=query,
                    ldatm=ldatm,
                    ld_func=ld_func,
                    ld_coeffs=ld_coeffs,
                    intens_weighting=intens_weighting,
                    ld_extrapolation_method=ld_extrapolation_method
                ).get_interpolated_values()

            return InterpResult(interps=10**log_inorms / ldint)

        elif atm.name == 'extern_planckint':
            if intens_weighting == 'photon':
                raise ValueError(f'the combination of atm={atm} and intens_weighting={intens_weighting} is not supported.')
            # TODO: add all other exceptions

            # wd_planckint() is rigid about its input, that's why we name a column:
            log10_inorms = libphoebe.wd_planckint(
                np.ascontiguousarray(query.pts[:, query.index('teffs')]),
                self.extern_wd_idx,
                self.wd_data["planck_table"]
            ).reshape(-1, 1)
            inorms = 10**(log10_inorms - 1)  # -1 is for cgs -> SI

            if ldint is None:
                ldint = self.interpolate_ldints(
                    query=query,
                    ldatm=ldatm,
                    ld_func=ld_func,
                    ld_coeffs=ld_coeffs,
                    intens_weighting=intens_weighting,
                    ld_extrapolation_method=ld_extrapolation_method
                ).get_interpolated_values()

            return InterpResult(interps=inorms/ldint)

        elif atm.name == 'extern_atmx':
            if intens_weighting == 'photon':
                raise ValueError(f'the combination of atm={atm} and intens_weighting={intens_weighting} is not supported.')
            # TODO: add all other exceptions

            log10_inorms = libphoebe.wd_atmint(
                np.ascontiguousarray(query.pts[:, query.index('teffs')]),
                np.ascontiguousarray(query.pts[:, query.index('loggs')]),
                np.ascontiguousarray(query.pts[:, query.index('abuns')]),
                self.extern_wd_idx,
                self.wd_data["planck_table"],
                self.wd_data["atm_table"]
            ).reshape(-1, 1)

            inorms = 10**(log10_inorms - 1)  # -1 for cgs -> metric
            return InterpResult(interps=inorms)

        elif atm.name in self.ndp.keys():  # atm in one of the model atmospheres
            result = InterpResult.from_ndpolator(
                self.ndp[atm.name].ndpolate(
                    f'inorm@{intens_weighting}',
                    query.subset(atm.basic_axis_names).pts,
                    extrapolation_method=atm_extrapolation_method
                )
            )

            log10ints = result.get_interpolated_values()
            dists = result.get_distances()

            if blending_method == 'blackbody' and dists is not None and np.any(dists > dist_threshold):
                if ld_extrapolation_method == 'none' and ld_func == 'interp':
                    raise ValueError('ld_extrapolation_method cannot be "none" when ld_func="interp" and blending_method="blackbody".')

                # TODO: it would make sense to use this only on the off-grid points!
                ints_bb = self.interpolate_inorms(
                    query=query,
                    atm=models.BlackbodyModelAtmosphere,
                    ldatm=ldatm,
                    ldint=ldint,
                    ld_func=ld_func,
                    ld_coeffs=ld_coeffs,
                    intens_weighting=intens_weighting,
                    atm_extrapolation_method=atm_extrapolation_method,
                    ld_extrapolation_method=ld_extrapolation_method
                ).get_interpolated_values()

                log10ints_bb = np.log10(ints_bb)

                off_grid = dists > dist_threshold

                log10ints_blended = log10ints.copy()
                log10ints_blended[off_grid] = (np.minimum(dists[off_grid], blending_margin) * log10ints_bb[off_grid] + np.maximum(blending_margin-dists[off_grid], 0) * log10ints[off_grid])/blending_margin
                blending_factors = np.minimum(dists, blending_margin)/blending_margin

                intensities = 10**log10ints_blended
                
                return InterpResult(interps=intensities, dists=dists, bfs=blending_factors)
            else:
                intensities = 10**log10ints
                return InterpResult(interps=intensities)

        else:
            raise ValueError(f'atm={atm.name} is not supported.')

    def interpolate_imus(self, query, atm=models.CK2004ModelAtmosphere, ldatm=models.CK2004ModelAtmosphere, ldint=None, ld_func='interp', ld_coeffs=None, intens_weighting='photon', atm_extrapolation_method='none', ld_extrapolation_method='none', blending_method='none', dist_threshold=1e-5, blending_margin=3):
        r"""
        Computes specific emergent passband intensities.

        Arguments
        ----------
        * `query` (<InterpQuery>, required): the interpolation query object.
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

        if f'{atm.name}:Imu' not in self.content:
            # external WD atmospheres are an exception (only Inorms, no Imus):
            if atm.name in ['extern_planckint', 'extern_atmx', 'blackbody'] and f'{atm.name}:Inorm' in self.content:
                pass
            else:
                raise ValueError(f'atm={atm.name} tables are not available in the {self.pbset}:{self.pbname} passband.')

        # let's first handle explicitly provided limb darkening func/coeffs;
        # in that case, imus = ld_mus * inorms:
        if ld_func != 'interp':
            if ld_coeffs is None:
                # interpolate limb darkening coefficients from the table:
                ld_coeffs = self.interpolate_ldcoeffs(
                    query=query,
                    ldatm=ldatm,
                    ld_func=ld_func,
                    intens_weighting=intens_weighting,
                    ld_extrapolation_method=ld_extrapolation_method
                ).get_interpolated_values()

            inorms = self.interpolate_inorms(
                query=query,
                atm=atm,
                ldatm=ldatm,
                ldint=ldint,
                ld_func=ld_func,
                ld_coeffs=ld_coeffs,
                intens_weighting=intens_weighting,
                atm_extrapolation_method=atm_extrapolation_method,
                ld_extrapolation_method=ld_extrapolation_method
            ).get_interpolated_values()

            ld_mus = self.ld_func(mu=query.pts[:, query.index('mus')], ld_coeffs=ld_coeffs, ld_func=ld_func).reshape(-1, 1)

            return InterpResult(interps=ld_mus * inorms)

        # now we need to handle the case of interpolated limb darkening:
        if atm.name == 'blackbody':
            if not hasattr(ldatm, 'mus'):
                raise ValueError(f'atm={atm.name} and ld_func={ld_func} are incompatible with ldatm={ldatm.name}.')

            # blackbody atmospheres are not limb-darkened (i.e., no specific
            # emergent passband intensities), so we need to appropriate ldatm's
            # limb darkening to normal emergent blackbody intensities:
            # 
            #   Imu^bb = Lmu Inorm^bb = Imu^atm / Inorm^atm * Inorm^bb

            inorms_bb = self.interpolate_inorms(
                query=query,
                atm=atm,
                ldatm=ldatm,
                ldint=ldint,
                ld_func=ld_func,
                ld_coeffs=ld_coeffs,
                intens_weighting=intens_weighting,
                atm_extrapolation_method=atm_extrapolation_method,
                ld_extrapolation_method=ld_extrapolation_method
            ).get_interpolated_values()

            result = self.interpolate_imus(
                query=query,
                atm=ldatm,
                ldatm=ldatm,
                ldint=ldint,
                ld_func=ld_func,
                ld_coeffs=ld_coeffs,
                intens_weighting=intens_weighting,
                atm_extrapolation_method=atm_extrapolation_method,
                ld_extrapolation_method=ld_extrapolation_method
            )

            imus_atm = result.get_interpolated_values()
            imus_dists = result.get_distances()

            inorms_atm = self.interpolate_inorms(
                query=query.subset(ldatm.basic_axis_names),
                atm=ldatm,
                ldatm=ldatm,
                ldint=ldint,
                ld_func=ld_func,
                ld_coeffs=ld_coeffs,
                intens_weighting=intens_weighting,
                atm_extrapolation_method=atm_extrapolation_method,
                ld_extrapolation_method=ld_extrapolation_method
            ).get_interpolated_values()

            imus_bb = imus_atm / inorms_atm * inorms_bb

            return InterpResult(interps=imus_bb, dists=imus_dists)

        elif atm.name in ['extern_planckint', 'extern_atmx']:
            raise ValueError(f'atm={atm.name} is incompatible with specific emergent intensities.')

        elif atm.name in self.ndp.keys():  # atm in one of the model atmospheres
            result = InterpResult.from_ndpolator(
                self.ndp[atm.name].ndpolate(
                    f'imu@{intens_weighting}',
                    query_pts=query.subset(atm.basic_axis_names + ['mus']).pts,
                    extrapolation_method=atm_extrapolation_method
                )
            )

            log10imus = result.get_interpolated_values()
            dists = result.get_distances()

            if blending_method == 'blackbody' and dists is not None and np.any(dists > dist_threshold):
                if ld_extrapolation_method == 'none' and ld_func == 'interp':
                    raise ValueError('ld_extrapolation_method cannot be "none" when ld_func="interp" and blending_method="blackbody".')

                imus_bb = self.interpolate_imus(
                    query=query,
                    atm=models.BlackbodyModelAtmosphere,
                    ldatm=ldatm,
                    ldint=ldint,
                    ld_func=ld_func,
                    ld_coeffs=ld_coeffs,
                    intens_weighting=intens_weighting,
                    atm_extrapolation_method=atm_extrapolation_method,
                    ld_extrapolation_method=ld_extrapolation_method
                ).get_interpolated_values()

                # Convert blackbody intensities to log10 for consistent blending
                log10imus_bb = np.log10(imus_bb)

                off_grid = dists > dist_threshold

                log10imus_blended = log10imus.copy()
                log10imus_blended[off_grid] = (
                    np.minimum(dists[off_grid], blending_margin) * log10imus_bb[off_grid] +
                    np.maximum(blending_margin-dists[off_grid], 0) * log10imus[off_grid]
                ) / blending_margin
                blending_factors = np.minimum(dists, blending_margin)/blending_margin

                intensities = 10**log10imus_blended

                return InterpResult(interps=intensities, dists=dists, bfs=blending_factors)

            else:
                intensities = 10**log10imus
                return InterpResult(interps=intensities)

        else:
            raise ValueError(f'atm={atm.name} is not supported.')

    def interpolate_ldints(self, query, ldatm=models.CK2004ModelAtmosphere, ld_func='linear', ld_coeffs=np.array([[0.5]]), intens_weighting='photon', ld_extrapolation_method='none'):
        """
        Computes ldint value for the given `ld_func` and `ld_coeffs`.

        Arguments
        ----------
        * `query` (<InterpQuery>, required): the interpolation query object.
        * `ldatm` (<models.ModelAtmosphere> subclass, optional,
          default=<models.CK2004ModelAtmosphere>): model atmosphere for
            limb darkening coefficients
        * `ld_func` (string, optional, default='linear'): limb darkening
          function
        * `ld_coeffs` (array, optional, default=[[0.5]]): limb darkening
          coefficients
        * `intens_weighting` (string, optional, default='photon'): intensity
          weighting mode
        * `ld_extrapolation_mode` (string, optional, default='none'): limb darkening
          extrapolation mode

        Returns
        -------
        * (array) ldint value(s)
        """

        # the most common scenario: interpolating ldints from the tables:
        if ld_func == 'interp':
            # ldints depend only on basic ldatm axes:
            ndp_output = self.ndp[ldatm.name].ndpolate(
                f'ldint@{intens_weighting}',
                query_pts=query.subset(ldatm.basic_axis_names).pts,
                extrapolation_method=ld_extrapolation_method)
            return InterpResult.from_ndpolator(ndp_output)

        # if LD coefficients are provided, make sure they are the correct shape:
        if ld_coeffs is not None:
            ld_coeffs = np.atleast_2d(ld_coeffs)

        # if LD coefficients are not provided and ld_func is not 'interp',
        # we can then get them from the tables:
        if ld_coeffs is None:
            ld_coeffs = self.interpolate_ldcoeffs(
                query=query,
                ldatm=ldatm,
                ld_func=ld_func,
                intens_weighting=intens_weighting,
                ld_extrapolation_method=ld_extrapolation_method
            ).get_interpolated_values()

        # now we have ld_coeffs and can compute ldints for any non-interp ld_func:
        ldints = np.ones(shape=(len(query.pts), 1))

        if ld_func == 'linear':
            ldints[:, 0] *= 1-ld_coeffs[:, 0]/3
        elif ld_func == 'logarithmic':
            ldints[:, 0] *= 1-ld_coeffs[:, 0]/3+2.*ld_coeffs[:, 1]/9
        elif ld_func == 'square_root':
            ldints[:, 0] *= 1-ld_coeffs[:, 0]/3-ld_coeffs[:, 1]/5
        elif ld_func == 'quadratic':
            ldints[:, 0] *= 1-ld_coeffs[:, 0]/3-ld_coeffs[:, 1]/6
        elif ld_func == 'power':
            ldints[:, 0] *= 1-ld_coeffs[:, 0]/5-ld_coeffs[:, 1]/3-3.*ld_coeffs[:, 2]/7-ld_coeffs[:, 3]/2
        else:
            raise ValueError(f'ld_func={ld_func} is not recognized.')

        return InterpResult(interps=ldints)

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
    the tables server (https://tables.phoebe-project.org by default).

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
        msg = "connection to online passbands at {} could not be established.  Check your internet connection or try again later.  If the problem persists and you're using a Mac, you may need to update openssl (see https://phoebe-project.org/help/faq).".format(_url_tables_server)
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
    the tables server (https://tables.phoebe-project.org by default), retrieving
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
    the tables server (https://tables.phoebe-project.org by default), retrieving
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
    the tables server (https://tables.phoebe-project.org by default).

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
                msg = "Connection to online passbands at {} could not be established.  Check your internet connection or try again later (can manually call phoebe.list_online_passbands(refresh=True) to retry).  If the problem persists and you're using a Mac, you may need to update openssl (see https://phoebe-project.org/help/faq).".format(_url_tables_server)
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
    the tables server (https://tables.phoebe-project.org by default)
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

    for atm in ['ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO', 'tremblay']:
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

    for atm in ['ck2004', 'phoenix', 'tmap_sdO', 'tmap_DA', 'tmap_DAO', 'tmap_DO', 'tremblay']:
        pb.compute_intensities(atm=atm, path=f'tables/{atm}', verbose=True)

    pb.import_wd_atmcof('tables/wd/atmcofplanck.dat', 'tables/wd/atmcof.dat', 7)

    pb.save('johnson_v.fits')
