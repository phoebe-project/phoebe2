import numpy as np
import astropy.units as u

try:
  basestring = basestring
except NameError:
  basestring = str

class Alias(object):
    def __init__(self, dict):
        self._dict = dict

    def map(self, key):
        return self._dict.get(key, key)


# https://matplotlib.org/examples/color/named_colors.html
# technically some of these (y, c, and m) don't exactly match, but as they "mean"
# the same thing, we'll handle the alias anyways
# Convention: always map to the spelled name and gray over grey
coloralias = Alias({'k': 'black', 'dimgrey': 'dimgray', 'grey': 'gray',
              'darkgrey': 'darkgray', 'lightgrey': 'lightgray',
               'w': 'white', 'r': 'red', 'y': 'yellow', 'g': 'green',
               'c': 'cyan', 'b': 'blue', 'm': 'magenta'})

# Convention: always map to the spelled name
linestylealias = Alias({'_': 'solid', '--': 'dashed', ':': 'dotted',
                      '-.': 'dashdot'})

class Group(object):
    def __init__(self, baseclass, reprlist, items):
        if not isinstance(items, list):
            raise TypeError("items must be of type list of {} objects".format(baseclass.__name__))

        # handle the case of unpacking nested groups (NOTE: only 1 deep)
        items_flatten = []
        for item in items:
            if isinstance(item, Group):
                for groupitem in item._items:
                    items_flatten.append(groupitem)
            else:
                items_flatten.append(item)

        for item in items_flatten:
            if not isinstance(item, baseclass):
                raise TypeError("each item in items must be of type {}".format(baseclass.__name__))

        self._items = items_flatten
        self._reprlist = reprlist

    def __repr__(self):
        info = " | ".join(["{}s: {}".format(attr,
                                           ", ".join(getattr(self, attr)))
                           for attr in self._reprlist])

        return "<{} | {} items | {}>".format(self.__class__.__name__, len(self), info)

    def __getitem__(self, ind):
        return self._items.__getitem__(ind)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def _get_attrs(self, attr):
        return [getattr(d, attr) for d in self._items]

    def _set_attrs(self, attr, value):
        for d in self._items:
            setattr(d, attr, value)


def _convert_unit(unit):
    if unit is None:
        unit = u.dimensionless_unscaled

    if isinstance(unit, basestring):
        unit = u.Unit(unit)

    if not (isinstance(unit, u.Unit) or isinstance(unit, u.CompositeUnit) or isinstance(unit, u.IrreducibleUnit)):
        raise TypeError("unit must be of type Unit, got: {}".format(unit))

    return unit

def tolist(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    else:
        return [value]

dimensions = ['i', 'x', 'y', 'z', 's', 'c']

global _inline
_inline = False
