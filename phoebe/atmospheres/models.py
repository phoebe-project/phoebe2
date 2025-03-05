import numpy as np
import glob
import os
import re


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
    * `basic_axis_names` (list): names of the basic axes; basic axes are
        axes that span the basic n-dimensional model atmosphere grid. The grid
        can be sparsely populated, but it must be regular.
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

    When a model atmosphere is instantiated via the from_path() method, the basic
    axes are populated with unique values from the filenames of the atmosphere
    fits files. The axes are then exported as numpy arrays.

    Attributes that are automatically populated:

    * `basic_axes` (tuple): tuple of numpy arrays for basic axes

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
    * `basic_axes` (tuple of ndarrays): values of the basic axes
    * `from_path` (bool): if True, the class is instantiated from a path

    Raises
    -------
    * `FileNotFoundError`: if the path does not exist
    """

    name = None
    prefix = None
    external = False

    # default axes:
    basic_axis_names = ['teffs', 'loggs', 'abuns']

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
            basic_axes = tuple([getattr(self, axis_name) for axis_name in self.basic_axis_names])
        else:
            if basic_axes is None:
                raise ValueError('basic_axes must be defined.')

        self.basic_axes = basic_axes

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def register(self):
        """
        Registers the model atmosphere with the global model table.
        """

        global _atmtable
        _atmtable.append(self.__class__)

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
            self.wls= np.load(os.path.join(path,wls_file)) 

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
        self.basic_axes = tuple([np.unique(getattr(self, name)) for name in self.basic_axis_names])

        # store all node indices:
        nodes = np.vstack([getattr(self, name) for name in self.basic_axis_names]).T
        self.indices = np.empty_like(nodes, dtype=int)
        for i, basic_axis in enumerate(self.basic_axes):
            self.indices[:, i] = np.searchsorted(basic_axis, nodes[:, i])

        return self

    def parse_rules(self, relative_filename):
        """
        Provides rules for parsing atmosphere fits files containing data.
        Only derived classes should implement this method.
        """

        return NotImplementedError

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
            new_axes = list(self.basic_axes)
            axis_index = self.basic_axis_names.index(axis_name)
            axis = self.basic_axes[axis_index]
            if axis_node not in axis:
                axis = np.append(axis, axis_node)
                axis.sort()

                new_axes[axis_index] = axis
                self.basic_axes = tuple(new_axes)
                nodes = np.vstack([getattr(self, name) for name in self.basic_axis_names]).T
                self.indices = np.empty_like(nodes, dtype=int)
                for i, basic_axis in enumerate(self.basic_axes):
                    self.indices[:, i] = np.searchsorted(basic_axis, nodes[:, i])
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
    basic_axis_names = ['teffs']
    external = True


class WDKurucz93ModelAtmosphere(ModelAtmosphere):
    """
    Wilson-Devinney Kurucz 1993 model atmosphere.
    """

    name = 'extern_atmx'
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

    basic_axis_names = ['teffs']
    teffs = np.logspace(2.5, 5.7, 500)  # this corresponds to the 316K-501187K range.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def limb_treatment(self, intensities):
        return intensities

    def intensity(self, wls):
        """
        Computes blackbody intensities.

        Arguments
        ----------
        * `wls` (array): wavelengths

        Returns
        --------
        * an array of intensities.
        """

        return 2 * 6.62607015e-34 * 2.99792458e8**2 / wls**5 / (np.exp(6.62607015e-34 * 2.99792458e8 / (wls * 1.380649e-23 * self.teffs[:, None])) - 1)


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

    mus = np.array([
        0., 0.001, 0.002, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04,
        0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
        0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.
    ])
    wls = np.arange(900., 39999.501, 0.5)/1e10  # AA -> m
    units = 1e7  # erg/s/cm^2/A -> W/m^3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    mus = np.array([
        0., 0.001, 0.002, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04,
        0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
        0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.
    ])
    wls = np.arange(500., 26000.)/1e10  # AA -> m
    units = 1  # W/m^3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    basic_axis_names = ['teffs', 'loggs']

    mus = np.array([
        0., 0.0034357 , 0.01801404, 0.04388279, 0.08044151, 0.12683405, 
        0.18197316, 0.2445665 , 0.31314696, 0.38610707, 0.46173674, 
        0.53826326, 0.61389293, 0.68685304, 0.7554335 , 0.81802684, 
        0.87316595, 0.91955849, 0.95611721, 0.98198596, 0.9965643 , 1.
    ])

    units = 1  # W/m^3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    basic_axis_names = ['teffs', 'loggs']

    mus = np.array([
        0., 0.00136799, 0.00719419, 0.01761889, 0.03254691, 0.05183939, 0.07531619,
        0.10275816, 0.13390887, 0.16847785, 0.20614219, 0.24655013, 0.28932435,
        0.33406564, 0.38035639, 0.42776398, 0.47584619, 0.52415388, 0.57223605,
        0.6196437, 0.66593427, 0.71067559, 0.75344991, 0.79385786, 0.83152216,
        0.86609102, 0.89724188, 0.92468378, 0.9481606,  0.96745302, 0.98238112,
        0.99280576, 0.99863193, 1.
    ])

    units = 1  # W/m^3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    basic_axis_names = ['teffs', 'loggs']

    mus = np.array([
        0., 0.00136799, 0.00719419, 0.01761889, 0.03254691, 0.05183939, 0.07531619,
        0.10275816, 0.13390887, 0.16847785, 0.20614219, 0.24655013, 0.28932435,
        0.33406564, 0.38035639, 0.42776398, 0.47584619, 0.52415388, 0.57223605,
        0.6196437, 0.66593427, 0.71067559, 0.75344991, 0.79385786, 0.83152216,
        0.86609102, 0.89724188, 0.92468378, 0.9481606,  0.96745302, 0.98238112,
        0.99280576, 0.99863193, 1.
    ])

    units = 1  # W/m^3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_rules(self, relative_filename):
        pars = re.split('[TGA.]+', relative_filename)
        return [
            float(pars[1]),     # teff
            float(pars[2])/100  # logg
        ]


class TMAPsdOModelAtmosphere(ModelAtmosphere):
    name = 'tmap_sdO'
    prefix = 'ts'
    basic_axis_names = ['teffs', 'loggs', 'abuns']

    mus = np.array([0., 0.00136799, 0.00719419, 0.01761889, 0.03254691, 
                    0.05183939, 0.07531619, 0.10275816, 0.13390887, 0.16847785,
                    0.20614219, 0.24655013, 0.28932435, 0.33406564, 0.38035639,
                    0.42776398, 0.47584619, 0.52415388, 0.57223605, 0.6196437,
                    0.66593427, 0.71067559, 0.75344991, 0.79385786, 0.83152216,
                    0.86609102, 0.89724188, 0.92468378, 0.9481606,  0.96745302,
                    0.98238112, 0.99280576, 0.99863193, 1.])
    units = 1  # W/m^3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_rules(self, relative_filename):
        pars = re.split('[TGA.]+', relative_filename)
        return [
            float(pars[1]),      # teff
            float(pars[2])/100,  # logg
            float(pars[3])/100   # abun
        ]


class TMAPDAOModelAtmosphere(ModelAtmosphere):
    name = 'tmap_DAO'
    prefix = 'tm'
    basic_axis_names = ['teffs', 'loggs', 'abuns']

    mus = np.array([0., 0.00136799, 0.00719419, 0.01761889, 0.03254691, 
                    0.05183939, 0.07531619, 0.10275816, 0.13390887, 0.16847785,
                    0.20614219, 0.24655013, 0.28932435, 0.33406564, 0.38035639,
                    0.42776398, 0.47584619, 0.52415388, 0.57223605, 0.6196437,
                    0.66593427, 0.71067559, 0.75344991, 0.79385786, 0.83152216,
                    0.86609102, 0.89724188, 0.92468378, 0.9481606,  0.96745302,
                    0.98238112, 0.99280576, 0.99863193, 1.])
    units = 1  # W/m^3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_rules(self, relative_filename):
        pars = re.split('[TGA.]+', relative_filename)
        return [
            float(pars[1]),      # teff
            float(pars[2])/100,  # logg
            float(pars[3])/100   # abun
        ]


# global model atmosphere table:
_atmtable = ModelAtmosphere.__subclasses__()


def atm_from_name(name):
    """
    Returns a model atmosphere class from its name.

    Arguments
    ----------
    * `name` (string): name of the model atmosphere

    Returns
    --------
    * the model atmosphere class
    """

    for atm in _atmtable:
        if atm.name == name:
            return atm

    raise ValueError(f'Atmosphere named "{name}" not found.')
