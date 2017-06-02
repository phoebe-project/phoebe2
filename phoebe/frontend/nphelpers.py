import numpy as np

from phoebe import u


class Arange(object):
    def __init__(self, start, stop, step, unit=None):
        self._start = float(start)
        self._stop = float(stop)
        self._step = float(step)
        self._unit = unit

    def __repr__(self):
        return "<Arange start={} stop(exclusive)={} step={} unit={}>".format(self.start, self.stop, self.step, self.unit)

    def __mul__(self, other):
        if not (isinstance(other, u.Unit) or isinstance(other, u.CompositeUnit) or isinstance(other, u.IrreducibleUnit)):
            raise ValueError("only support multiplication with units")

        return Arange(self.start, self.stop, self.step, unit=other)

    def to_json(self):
        return {'nphelper': 'Arange', 'start': self.start, 'stop': self.stop, 'step': self.step}

    @property
    def start(self):
        return self._start

    def set_start(self, start):
        self._start = float(start)

    @property
    def stop(self):
        return self._stop

    def set_stop(self, stop):
        self._stop = float(stop)

    @property
    def step(self):
        return self._step

    def set_step(self, step):
        self._step = float(step)

    @property
    def unit(self):
        return self._unit

    def to(self, unit):
        start = (self.start*self.unit).to(unit).value
        stop = (self.stop*self.unit).to(unit).value
        step = (self.step*self.unit).to(unit).value

        return Arange(start, stop, step, unit=unit)

    def to_array(self):
        arr = np.arange(self.start, self.stop, self.step)
        if self.unit:
            return arr*self.unit
        else:
            return arr

    def to_linspace(self):
        num = (self.stop-self.start)/(self.step)
        return Linspace(self.start, self.stop-self.step, num)

class Linspace(object):
    def __init__(self, start, stop, num, unit=None):
        self._start = float(start)
        self._stop = float(stop)
        self._num = int(num)
        self._unit = unit

    def __repr__(self):
        return "<Linspace start={} stop(inclusive)={} num={} unit={}>".format(self.start, self.stop, self.num, self.unit)

    def __mul__(self, other):
        if not (isinstance(other, u.Unit) or isinstance(other, u.CompositeUnit) or isinstance(other, u.IrreducibleUnit)):
            raise ValueError("only support multiplication with units")

        return Linspace(self.start, self.stop, self.num, unit=other)

    def to_json(self):
        return {'nphelper': 'Linspace', 'start': self.start, 'stop': self.stop, 'num': self.num}

    @property
    def start(self):
        return self._start

    def set_start(self, start):
        self._start = float(start)

    @property
    def stop(self):
        return self._stop

    def set_stop(self, stop):
        self._stop = float(stop)

    @property
    def num(self):
        return self._num

    def set_num(self, num):
        self._num = int(num)

    @property
    def unit(self):
        return self._unit

    def to(self, unit):
        start = (self.start*self.unit).to(unit).value
        stop = (self.stop*self.unit).to(unit).value

        return Linspace(start, stop, self.num, unit=unit)

    def to_array(self):
        arr = np.linspace(self.start, self.stop, self.num)
        if self.unit:
            return arr*self.unit
        else:
            return arr

    def to_arange(self):
        step = (self.stop-self.start)/(self.num-1)
        return Arange(self.start, self.stop+step, step)

def from_json(json_dict):
    helper = json_dict['nphelper']
    if helper == 'Linspace':
        return Linspace(start=json_dict['start'], stop=json_dict['stop'], num=json_dict['num'])
    elif helper == 'Arange':
        return Arange(start=json_dict['start'], stop=json_dict['stop'], step=json_dict['step'])
    else:
        return None
