import numpy as np
import astropy.units as u

import sys
import os

if sys.version_info[0] < 3:
    from urllib2 import urlopen as _urlopen
else:
    from urllib.request import urlopen as _urlopen

import json as _json

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

        self._baseclass = baseclass
        self._items = items_flatten
        self._reprlist = reprlist

    def __repr__(self):
        info = " | ".join(["{}s: {}".format(attr,
                                           ", ".join(getattr(self, attr)))
                           for attr in self._reprlist])

        return "<{} | {} items | {}>".format(self.__class__.__name__, len(self), info)

    # @classmethod
    # def from_dict(cls, dict):
    #     return cls(**dict)
    #
    # def to_dict(self):
    #     return {'baseclass': self._baseclass,
    #             'items': self._items,
    #             'reprlist': self._reprlist}

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

def arraytolistrecursive(value):
    if hasattr(value, '__iter__') and not isinstance(value, str):
        return [arraytolistrecursive(v) for v in value]
    else:
        return value

def _bytes(s):
    if sys.version_info[0] == 3:
        return bytes(s, 'utf-8')
    else:
        return bytes(s)

def _json_safe(item):
    if isinstance(item, np.ndarray):
        return arraytolistrecursive(item)
    elif isinstance(item, dict):
        return {k: _json_safe(v) for k,v in item.items()}
    else:
        return item

def _parse_json(pairs):
    """
    modified from:
    https://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-from-json#34796078

    pass this to the object_pairs_hook kwarg of json.load/loads
    """
    def _string(item):
        if isinstance(item, bytes):
            # return item.decode('utf-8')
            return _bytes(item)
        elif sys.version_info[0] == 2 and isinstance(item, unicode):
            return item.encode('utf-8')
        else:
            return item

    new_pairs = []
    for key, value in pairs:
        key = _string(key)

        if isinstance(value, dict):
            value = _parse_json(value.items())
        elif isinstance(value, list):
            value = [_string(v) for v in value]
        else:
            value = _string(value)

        new_pairs.append((key, value))
    return dict(new_pairs)

def save(dict, filename):
    filename = os.path.expanduser(filename)
    f = open(filename, 'w')
    _json.dump(dict, f,
              sort_keys=False, indent=0)

    f.close()

    return filename

def load(filename):
    if filename[:4] == 'http':
        resp = _urlopen(filename)
        dict = _json.loads(resp.read(), object_pairs_hook=_parse_json)
    else:
        filename = os.path.expanduser(filename)
        f = open(filename, 'r')
        dict = _json.load(f, object_pairs_hook=_parse_json)
        f.close()

    return dict

dimensions = ['i', 'x', 'y', 'z', 's', 'c']

global _inline
_inline = False
