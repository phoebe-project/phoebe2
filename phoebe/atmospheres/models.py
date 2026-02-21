import numpy as np
import glob
import os
import re
from phoebe import c


class ModelAtmosphere:
    """
    A parent class for handling model atmosphere data. Please note that only
    derived classes should be instantiated.

    Model atmospheres are approximations to stellar atmospheres. Each model
    connects input parameters (for example, effective temperature, surface
    gravity, and chemical abundances) to output parameters (for example,
    intensities at different wavelengths and angles). The ModelAtmosphere
    class provides a common interface for handling different model
    atmospheres.

    In order to have a model atmosphere supported, the following attributes
    need to be defined:

    * `name` (string): name of the model atmosphere
    * `prefix` (string): prefix for the model atmosphere fits keywords
    * `basic_axes` (dict): names/values of the basic axes; basic axes are
        axes that span the basic n-dimensional model atmosphere grid. The grid
        can be sparsely populated, but it must be regular.
    * `assumed_axes` (dict): names/values of the assumed axes; assumed axes
        are single value axes that are not tabulated but are assumed to have
        a fixed value across the entire grid. PHOEBE will automatically check
        if the parameter matches the assumed value for a given model and raise
        a warning if it does not.
    * `mus` (array): specific angles, mu=cos(theta), where theta is the angle
        between the observer and the surface normal.
    * `wls` (array): wavelengths of the model atmosphere intensities.
    * `units` (float): intensity unit conversion factor. The intensities are
        usually given in erg/s/cm^2/A, which is converted to W/m^3 by
        multiplying with this factor.

    In addition, the following methods need to or may be overloaded:

    * `parse_rules`: provides rules for parsing atmosphere fits filenames to
        extract basic axis values.
    * `limb_treatment`: defines how intensities at the exact limb (mu=0) should
        be treated. By default, the intensities are linearly extrapolated to
        mu=0.

    When a model atmosphere is instantiated via the from_path() method, basic
    axes are populated with unique values from the filenames of the atmosphere
    fits files.

    Attributes that are automatically populated:

    * `basic_axis_names` (list): list of basic axis names
    * `ndp_basic_axes` (tuple): tuple of numpy arrays for basic axes
    * `axis_limits` (dict): dictionary of axis limits for all axes
        (both basic and assumed axes)

    In the case of from_path() instantiation, the following attributes are
    also populated:

    * `models` (list): list of atmosphere fits files
    * `nmodels` (int): number of atmosphere fits files
    * `indices` (array): array of indices for all defined nodes in the model
        atmosphere
    * `[axis_name]` (array): numpy array for each axis, where the name is
        automatically inferred from the basic axis names

    Arguments
    ----------
    * `basic_axes` (dict): names/values of the basic axes
    * `from_path` (bool): if True, the class is instantiated from a path

    Raises
    -------
    * `FileNotFoundError`: if the path does not exist
    """

    name = None
    prefix = None
    external = False

    basic_axes = {}
    assumed_axes = {}

    def __init__(self, basic_axes=None, from_path=False):
        if from_path:
            return

        self.path = None

        # the model needs to either provide tabulated intensities or
        # a function to compute them. If the model provides tabulated
        # intensities, basic_axes must be defined; if it provides a
        # function, basic_axes will be automatically determined from
        # axis definitions via class attributes.

        if hasattr(self, 'intensity') and callable(self.intensity):
            # the model provides a function to compute intensities.
            for axis_name in self.basic_axis_names:
                if not hasattr(self, axis_name):
                    raise ValueError(f'Model atmosphere named basic axis "{axis_name}" but it did not define it.')
            basic_axes = {axis_name: getattr(self, axis_name) for axis_name in self.basic_axis_names}
        else:
            if basic_axes is None:
                raise ValueError('basic_axes must be defined.')

        self.basic_axes = basic_axes

    def __init_subclass__(cls, **kwargs):
        """
        Automatically populate basic_axis_names and axis_limits from
        basic_axes.

        This method is called when a subclass is created and precomputes
        class-level attributes for efficient access:
        * basic_axis_names: list of axis names from basic_axes keys
        * axis_limits: dictionary of (min, max) tuples for each axis
        """
        super().__init_subclass__(**kwargs)
        if isinstance(cls.basic_axes, dict):
            cls.basic_axis_names = list(cls.basic_axes.keys())
            # precompute axis_limits as a class attribute for class-level access:
            limits = {}
            for name, arr in cls.basic_axes.items():
                if len(arr) > 0:
                    limits[name] = (arr[0], arr[-1])
            limits.update({name: (val, val) for name, val in cls.assumed_axes.items()})
            cls.axis_limits = limits

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def register(self):
        """
        Registers the model atmosphere with the global model table.
        """

        _atmtable[self.__class__.name] = self.__class__

    @property
    def ndp_basic_axes(self):
        """
        Returns the basic axes as a tuple of arrays, suitable for
        passing to ndpolator.
        """
        return tuple(self.basic_axes.values())

    @classmethod
    def has_axis(cls, axis_name):
        """
        Checks if the model atmosphere has the specified axis (either as
        an interpolated axis or fixed/assumed value)

        Arguments
        ----------
        * `axis_name` (string): name of the axis

        Returns
        --------
        * True if the axis exists, False otherwise.
        """
        return axis_name in cls.basic_axis_names or axis_name in cls.assumed_axes.keys()

    @classmethod
    def get_axis_limits(cls, axis):
        """
        Get the limits for a given axis.

        Arguments
        ---------
        * `axis` (string): name of the axis.

        Returns
        -------
        * (tuple) minimum and maximum values for the given axis,
          or (fixed_value, fixed_value) if the axis is fixed,
          or None if the axis exists but limits are unknown (e.g., external atmospheres).

        Raises
        ------
        * ValueError: if the axis is not found in the model atmosphere.
        """
        if axis in cls.axis_limits:
            return cls.axis_limits[axis]
        elif axis in cls.basic_axis_names:
            # Axis exists but has no limits (e.g., external atmosphere with empty array)
            return None
        else:
            raise ValueError(f"axis '{axis}' not found in model atmosphere '{cls.name}'")

    @classmethod
    def from_path(cls, path, wls_file=None):
        """
        Instantiates the class and all attributes from a given path.

        Arguments
        ----------
        * `path` (string): relative or absolute path to data files

        Returns
        --------
        * a new model atmosphere object
        """

        self = cls(from_path=True)
        self.path = path

        if wls_file is not None:
            self.wls = np.load(os.path.join(path, wls_file))

        try:
            self.models = glob.glob(os.path.join(path, '*fits'))
            self.nmodels = len(self.models)
        except FileNotFoundError:
            raise FileNotFoundError(f'path {path} does not exist.')

        # initialize arrays for basic axes:
        for name in self.basic_axis_names:
            setattr(self, name, np.empty(self.nmodels))

        # parse the filenames and populate the arrays:
        for i, model in enumerate(self.models):
            relative_filename = os.path.basename(model)
            basic_node_values = self.parse_rules(relative_filename)
            for j, name in enumerate(self.basic_axis_names):
                getattr(self, name)[i] = basic_node_values[j]

        # export basic axes:
        self.basic_axes = {name: np.unique(getattr(self, name)) for name in self.basic_axis_names}

        # store all node indices:
        self._recompute_indices()

        return self

    def parse_rules(self, relative_filename):
        """
        Provides rules for parsing atmosphere fits files containing data.
        Only derived classes should implement this method.
        """
        raise NotImplementedError

    def _recompute_indices(self):
        """
        Recomputes the indices array based on current basic axes.
        """
        nodes = np.vstack([getattr(self, name) for name in self.basic_axis_names]).T
        self.indices = np.empty_like(nodes, dtype=int)
        for i, basic_axis in enumerate(self.ndp_basic_axes):
            self.indices[:, i] = np.searchsorted(basic_axis, nodes[:, i])

    def add_axis_node(self, axis_name, axis_node):
        """
        Adds a node to the specified axis. This method is used
        when we want to add another value into the convex hull
        spun by the current axes. If you do not know why you would
        use this method, you likely should not use it.

        Arguments
        ----------
        * `axis_name` (string): name of the axis
        * `axis_node` (float): value of the node to be added
        """

        if axis_name in self.basic_axis_names:
            axis = self.basic_axes[axis_name]
            if axis_node not in axis:
                axis = np.append(axis, axis_node)
                axis.sort()

                self.basic_axes[axis_name] = axis
                self._recompute_indices()
        else:
            raise ValueError(f"Axis name '{axis_name}' not recognized.")

    def limb_treatment(self, intensities):
        """
        Define how intensities at the exact limb (mu=0) should be treated. By
        default, the intensities are linearly extrapolated to mu=0.

        Arguments
        ----------
        * `intensities` (array): intensities across all mus

        Returns
        --------
        * an array of intensities with the limb treatment applied.
        """

        intensities[0] = max(1e-12, intensities[1] + (intensities[2]-intensities[1])/(self.mus[2]-self.mus[1])*(self.mus[0]-self.mus[1]))
        return intensities


class WDBlackbodyModelAtmosphere(ModelAtmosphere):
    """
    Wilson-Devinney blackbody model atmosphere.
    """

    name = 'extern_planckint'
    basic_axes = {
        'teffs': np.array([])
    }
    external = True


class WDKurucz93ModelAtmosphere(ModelAtmosphere):
    """
    Wilson-Devinney Kurucz 1993 model atmosphere.
    """

    name = 'extern_atmx'
    basic_axes = {
        'teffs': np.array([]),
        'loggs': np.array([]),
        'abuns': np.array([])
    }
    external = True


class BlackbodyModelAtmosphere(ModelAtmosphere):
    """
    Blackbody model atmosphere.

    The blackbody model atmosphere is a simple model atmosphere that assumes
    the object is a blackbody. The grid is defined by a single axis,
    effective temperature (teffs).
    """

    name = 'blackbody'
    prefix = 'bb'

    teffs = np.logspace(2.5, 5.7, 500)  # this corresponds to the 316K-501187K range.

    basic_axes = {
        'teffs': teffs
    }

    def limb_treatment(self, intensities):
        return intensities

    def intensity(self, wls):
        """
        Computes blackbody intensities.

        Arguments
        ----------
        * `wls` (array): wavelengths in meters

        Returns
        --------
        * an array of intensities in W/m^3.
        """
        hc_over_kT = c.h.value * c.c.value / (wls * c.k_B.value * self.teffs[:, None])
        return 2 * c.h.value * c.c.value**2 / wls**5 / (np.exp(hc_over_kT) - 1)


class CK2004ModelAtmosphere(ModelAtmosphere):
    """
    Castelli & Kurucz (2004) model atmosphere.

    The CK2004 model atmosphere is a grid of model atmospheres computed by
    Castelli & Kurucz (2004). The grid is defined by effective temperature
    (teff), surface gravity (logg), and chemical abundance (abun). The
    intensities are computed for 37 angles (mus) on the 900-40000A wavelength
    range.
    """

    name = 'ck2004'
    prefix = 'ck'

    teffs = np.array([
        3500.,  3750.,  4000.,  4250.,  4500.,  4750.,  5000.,  5250.,
        5500.,  5750.,  6000.,  6250.,  6500.,  6750.,  7000.,  7250.,
        7500.,  7750.,  8000.,  8250.,  8500.,  8750.,  9000.,  9250.,
        9500.,  9750., 10000., 10250., 10500., 10750., 11000., 11250.,
       11500., 11750., 12000., 12250., 12500., 12750., 13000., 14000.,
       15000., 16000., 17000., 18000., 19000., 20000., 21000., 22000.,
       23000., 24000., 25000., 26000., 27000., 28000., 29000., 30000.,
       31000., 32000., 33000., 34000., 35000., 36000., 37000., 38000.,
       39000., 40000., 41000., 42000., 43000., 44000., 45000., 46000.,
       47000., 48000., 49000., 50000.
    ])

    loggs = np.array([
        0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.
    ])

    abuns = np.array([
        -2.5, -2., -1.5, -1., -0.5,  0.,  0.2,  0.5
    ])

    basic_axes = {
        'teffs': teffs,
        'loggs': loggs,
        'abuns': abuns
    }

    mus = np.array([
        0., 0.001, 0.002, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04,
        0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
        0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.
    ])
    wls = np.arange(900., 39999.501, 0.5)/1e10  # AA -> m
    units = 1e7  # erg/s/cm^2/A -> W/m^3

    def parse_rules(self, relative_filename):
        return [
            float(relative_filename[1:6]),  # teff
            float(relative_filename[7:9])/10,  # logg
            float(relative_filename[10:12])/10 * (-1 if relative_filename[9] == 'M' else 1)  # abun
        ]


class PhoenixModelAtmosphere(ModelAtmosphere):
    """
    Phoenix (Husser et al. 2012) model atmosphere.

    The Phoenix model atmosphere is a grid of model atmospheres computed by
    Husser et al. (2012). The grid is defined by effective temperature (teff),
    surface gravity (logg), and chemical abundance (abun). The intensities are
    computed for 37 angles between 500 and 26000 Angstroms.
    """

    name = 'phoenix'
    prefix = 'ph'

    teffs = np.array([
        2300.,  2400.,  2500.,  2600.,  2700.,  2800.,  2900.,  3000.,
        3100.,  3200.,  3300.,  3400.,  3500.,  3600.,  3700.,  3800.,
        3900.,  4000.,  4100.,  4200.,  4300.,  4400.,  4500.,  4600.,
        4700.,  4800.,  4900.,  5000.,  5100.,  5200.,  5300.,  5400.,
        5500.,  5600.,  5700.,  5800.,  5900.,  6000.,  6100.,  6200.,
        6300.,  6400.,  6500.,  6600.,  6700.,  6800.,  6900.,  7000.,
        7200.,  7400.,  7600.,  7800.,  8000.,  8200.,  8400.,  8600.,
        8800.,  9000.,  9200.,  9400.,  9600.,  9800., 10000., 10200.,
       10400., 10600., 10800., 11000., 11200., 11400., 11600., 11800.,
       12000.
    ])

    loggs = np.array([
        0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.
    ])

    abuns = np.array([
        -4., -3., -2., -1.5, -1., -0.5,  0.,  0.5,  1.
    ])

    basic_axes = {
        'teffs': teffs,
        'loggs': loggs,
        'abuns': abuns
    }

    mus = np.array([
        0., 0.001, 0.002, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04,
        0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
        0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.
    ])
    wls = np.arange(500., 26000.)/1e10  # AA -> m
    units = 1  # W/m^3

    def parse_rules(self, relative_filename):
        return [
            float(relative_filename[1:6]),  # teff
            float(relative_filename[7:11]),  # logg
            float(relative_filename[12:16])  # abun
        ]


class TremblayModelAtmosphere(ModelAtmosphere):
    """
    Tremblay DA model atmosphere.
    """

    name = 'tremblay'
    prefix = 'tr'

    teffs = np.array([
        3750.,  4000.,  4250.,  4500.,  4750.,  5000.,  5250.,  5500.,
        5750.,  6000.,  6250.,  6500.,  7000.,  7500.,  8000.,  8500.,
        9000.,  9500., 10000., 10500., 11000., 11500., 12000., 12500.,
       13000., 13500., 14000., 14500., 15000., 15500., 16000., 16500.,
       17000., 20000., 25000., 30000., 35000., 40000., 45000., 50000.,
       55000., 60000.
    ])

    loggs = np.array([
        5., 6., 7., 8., 9.
    ])

    basic_axes = {
        'teffs': teffs,
        'loggs': loggs
    }
    assumed_axes = {
        'loghefracs': -10.0
    }

    mus = np.array([
        0., 0.0034357 , 0.01801404, 0.04388279, 0.08044151, 0.12683405, 
        0.18197316, 0.2445665 , 0.31314696, 0.38610707, 0.46173674, 
        0.53826326, 0.61389293, 0.68685304, 0.7554335 , 0.81802684, 
        0.87316595, 0.91955849, 0.95611721, 0.98198596, 0.9965643 , 1.
    ])

    units = 1  # W/m^3

    def parse_rules(self, relative_filename):
        pars = re.split('[TGA.]+', relative_filename)
        return [
            float(pars[1]),  # teff
            float(pars[2])/100  # logg
        ]


class TMAPDOModelAtmosphere(ModelAtmosphere):
    """
    TMAP model atmosphere.
    """

    name = 'tmap_DO'
    prefix = 'to'

    teffs = np.array([
        40000.,  45000.,  50000.,  55000.,  60000.,  65000.,  70000.,
        75000.,  80000.,  85000.,  90000.,  95000., 100000., 110000.,
       120000., 130000., 140000., 150000., 160000., 170000., 180000.,
       190000., 200000.
    ])

    loggs = np.array([
        6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5
    ])

    basic_axes = {
        'teffs': teffs,
        'loggs': loggs
    }

    assumed_axes = {
        'loghefracs': 9.4
    }

    mus = np.array([
        0., 0.00136799, 0.00719419, 0.01761889, 0.03254691, 0.05183939, 0.07531619,
        0.10275816, 0.13390887, 0.16847785, 0.20614219, 0.24655013, 0.28932435,
        0.33406564, 0.38035639, 0.42776398, 0.47584619, 0.52415388, 0.57223605,
        0.6196437, 0.66593427, 0.71067559, 0.75344991, 0.79385786, 0.83152216,
        0.86609102, 0.89724188, 0.92468378, 0.9481606,  0.96745302, 0.98238112,
        0.99280576, 0.99863193, 1.
    ])

    units = 1  # W/m^3

    def parse_rules(self, relative_filename):
        pars = re.split('[TGA.]+', relative_filename)
        return [
            float(pars[1]),     # teff
            float(pars[2])/100  # logg
        ]


class TMAPDAModelAtmosphere(ModelAtmosphere):
    """
    TMAP model atmosphere.
    """

    name = 'tmap_DA'
    prefix = 'ta'

    teffs = np.array([
        20000.,  21000.,  22000.,  23000.,  24000.,  25000.,  26000.,
        27000.,  28000.,  29000.,  30000.,  31000.,  32000.,  33000.,
        34000.,  35000.,  36000.,  37000.,  38000.,  39000.,  40000.,
        45000.,  50000.,  55000.,  60000.,  65000.,  70000.,  75000.,
        80000.,  85000.,  90000.,  95000., 100000., 110000., 120000.,
       130000., 140000., 150000., 160000., 170000., 180000., 190000.,
       200000.
    ])

    loggs = np.array([
        6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5
    ])

    basic_axes = {
        'teffs': teffs,
        'loggs': loggs
    }
    assumed_axes = {
        'loghefracs': -10.0
    }

    mus = np.array([
        0., 0.00136799, 0.00719419, 0.01761889, 0.03254691, 0.05183939, 0.07531619,
        0.10275816, 0.13390887, 0.16847785, 0.20614219, 0.24655013, 0.28932435,
        0.33406564, 0.38035639, 0.42776398, 0.47584619, 0.52415388, 0.57223605,
        0.6196437, 0.66593427, 0.71067559, 0.75344991, 0.79385786, 0.83152216,
        0.86609102, 0.89724188, 0.92468378, 0.9481606,  0.96745302, 0.98238112,
        0.99280576, 0.99863193, 1.
    ])

    units = 1  # W/m^3

    def parse_rules(self, relative_filename):
        pars = re.split('[TGA.]+', relative_filename)
        return [
            float(pars[1]),     # teff
            float(pars[2])/100  # logg
        ]


class TMAPsdOModelAtmosphere(ModelAtmosphere):
    name = 'tmap_sdO'
    prefix = 'ts'

    teffs = np.array([
        40000.,  45000.,  50000.,  55000.,  60000.,  65000.,  70000.,
        75000.,  80000.,  85000.,  90000.,  95000., 100000., 105000.,
       110000., 115000., 120000., 125000., 130000., 135000., 140000.
    ])

    loggs = np.array([
        4.75, 5., 5.25, 5.5, 5.75, 6., 6.25, 6.5
    ])

    loghefracs = np.array([
        -1.55, -1.2, -0.97, -0.78, -0.6, -0.42
    ])

    basic_axes = {
        'teffs': teffs,
        'loggs': loggs,
        'loghefracs': loghefracs
    }

    mus = np.array([
        0., 0.00136799, 0.00719419, 0.01761889, 0.03254691,
        0.05183939, 0.07531619, 0.10275816, 0.13390887, 0.16847785,
        0.20614219, 0.24655013, 0.28932435, 0.33406564, 0.38035639,
        0.42776398, 0.47584619, 0.52415388, 0.57223605, 0.6196437,
        0.66593427, 0.71067559, 0.75344991, 0.79385786, 0.83152216,
        0.86609102, 0.89724188, 0.92468378, 0.9481606,  0.96745302,
        0.98238112, 0.99280576, 0.99863193, 1.
    ])
    units = 1  # W/m^3

    def parse_rules(self, relative_filename):
        pars = re.split('[TGA.]+', relative_filename)
        return [
            float(pars[1]),      # teff
            float(pars[2])/100,  # logg
            float(pars[3])/100   # loghefrac
        ]


class TMAPDAOModelAtmosphere(ModelAtmosphere):
    name = 'tmap_DAO'
    prefix = 'tm'

    teffs = np.array([
        40000.,  45000.,  50000.,  55000.,  60000.,  65000.,  70000.,
        75000.,  80000.,  85000.,  90000.,  95000., 100000., 110000.,
       120000., 130000., 140000., 150000., 160000., 170000., 180000.,
       190000., 200000.
    ])

    loggs = np.array([
        6. , 6.5, 7. , 7.5, 8. , 8.5, 9.
    ])

    loghefracs = np.array([
        -5., -4., -3., -2., -1.,  0.
    ])

    basic_axes = {
        'teffs': teffs,
        'loggs': loggs,
        'loghefracs': loghefracs
    }

    mus = np.array([
        0., 0.00136799, 0.00719419, 0.01761889, 0.03254691, 
        0.05183939, 0.07531619, 0.10275816, 0.13390887, 0.16847785,
        0.20614219, 0.24655013, 0.28932435, 0.33406564, 0.38035639,
        0.42776398, 0.47584619, 0.52415388, 0.57223605, 0.6196437,
        0.66593427, 0.71067559, 0.75344991, 0.79385786, 0.83152216,
        0.86609102, 0.89724188, 0.92468378, 0.9481606,  0.96745302,
        0.98238112, 0.99280576, 0.99863193, 1.
    ])
    units = 1  # W/m^3

    def parse_rules(self, relative_filename):
        pars = re.split('[TGA.]+', relative_filename)
        return [
            float(pars[1]),      # teff
            float(pars[2])/100,  # logg
            float(pars[3])/100   # loghefrac
        ]


# global model atmosphere table:(dict of name -> class):
_atmtable = {atm.name: atm for atm in ModelAtmosphere.__subclasses__()}

