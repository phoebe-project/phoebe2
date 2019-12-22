from matplotlib import colors, markers, cm
import matplotlib.pyplot as plt
from . import common

_mplcolors = ['black', 'blue', 'red', 'green']
_mplcolors += [common.coloralias.map(c) for c in list(colors.ColorConverter.colors.keys()) + list(colors.cnames.keys()) if common.coloralias.map(c) not in _mplcolors and 'xkcd' not in c and ':' not in c]
_mplmarkers = ['.', 'o', '+', 's', '*', 'x', 'v', '^', '<', '>', 'p', 'h', 'o', 'D']
# could do matplotlib.markers.MarkerStyle.markers.keys()
_mpllinestyles = ['solid', 'dashed', 'dotted', 'dashdot'] #, 'None']
# could also do matplotlib.lines.lineStyles.keys()

# https://matplotlib.org/examples/color/colormaps_reference.html
# unfortunately some colormaps have been added, but we want the nice ones
# first.  So let's define an order based on 1.5, remove any that aren't supported
# with the installed version of MPL and then add any that the installed version
# has that we missed.
_mplcmaps = ['viridis', 'plasma', 'inferno', 'magma',
                 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
             'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                    'hot', 'afmhot', 'gist_heat', 'copper',
             'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
             'Pastel1', 'Pastel2', 'Paired', 'Accent',
                    'Dark2', 'Set1', 'Set2', 'Set3',
                    'tab10', 'tab20', 'tab20b', 'tab20c',
             'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
                    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

_mplcmaps = [cm for cm in _mplcmaps if cm in plt.colormaps()]
_mplcmaps += [cm for cm in plt.colormaps() if cm not in _mplcmaps]



class MPLPropCycler(object):
    def __init__(self, prop, options=[]):
        self._prop = prop
        self._options_orig = options
        self._options = options
        self._used = []
        self._used_tmp = []

    def __repr__(self):
        return '<{}cycler | cycle: {} | used: {}>'.format(self._prop,
                                                          self.cycle,
                                                          self.used)


    @property
    def cycle(self):
        return self._options

    @cycle.setter
    def cycle(self, cycle):
        for option in cycle:
            if option not in self._options_orig:
                raise ValueError("invalid option: {}".format(option))
        self._options = cycle

    @property
    def used(self):
        return list(set(self._used + self._used_tmp))

    @property
    def next(self):
        for option in self.cycle:
            if option not in self.used:
                self.add_to_used(option)
                return option
        else:
            return self.cycle[-1]

    @property
    def next_tmp(self):
        for option in self.cycle:
            if option not in self.used:
                self.add_to_used_tmp(option)
                return option
        else:
            return self.cycle[-1]

    def get(self, option=None):
        if option is not None:
            self.add_to_used(option)
            return option
        else:
            return self.next

    def get_tmp(self, option=None):
        if option is not None:
            self.add_to_used_tmp(option)
            return option
        else:
            return self.next_tmp

    def check_validity(self, option):
        if option not in self._options_orig:
            raise ValueError("{} not one of {}".format(option, self._options_orig))

    def add_to_used(self, option):
        if option in [None, 'None', 'none', 'face']:
            return
        if option not in self._options_orig:
            return
            # raise ValueError("{} not one of {}".format(option, self._options_orig))
        if option not in self._used:
            self._used.append(option)

    def replace_used(self, oldoption, newoption):
        if newoption in [None, 'None', 'none']:
            return
        if newoption not in self._options_orig:
            raise ValueError("{} not one of {}".format(newoption, self._options_orig))

        if oldoption in self._used:
            ind = self._used.index(oldoption)
            self._used[ind] = newoption
        elif oldoption in self._used_tmp:
            # in many cases the old option may actually be in _used_tmp but
            # None will be passed because we don't have access to the previous
            # state of the color cycler.  But _used_tmp will be reset on the
            # next draw anyways, so this doesn't really hurt anything.
            self._used_tmp.remove(oldoption)
            self.add_to_used(newoption)
        else:
            self.add_to_used(newoption)


    def add_to_used_tmp(self, option):
        if option in [None, 'None', 'none']:
            return
        if option not in self._options_orig:
            raise ValueError("{} not one of {}".format(option, self._options_orig))
        if option not in self._used_tmp:
            self._used_tmp.append(option)

    def clear_tmp(self):
        self._used_tmp = []
        return



class MPLColorCycler(MPLPropCycler):
    def __init__(self):
        super(MPLColorCycler, self).__init__('color', options=_mplcolors)

class MPLCmapCycler(MPLPropCycler):
    def __init__(self):
        super(MPLCmapCycler, self).__init__('cmap', options=_mplcmaps)

class MPLMarkerCycler(MPLPropCycler):
    def __init__(self):
        super(MPLMarkerCycler, self).__init__('marker', options=_mplmarkers)

class MPLLinestyleCycler(MPLPropCycler):
    def __init__(self):
        super(MPLLinestyleCycler, self).__init__('linestyle', options=_mpllinestyles)
