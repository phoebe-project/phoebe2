from phoebe.parameters import *
import numpy as np

def _units_labels_lims(axis, default_unit, **kwargs):
    params = []

    params += [UnitParameter(qualifier='{}unit'.format(axis), value=kwargs.get('{}unit'.format(axis), default_unit), description='Unit for {}-axis'.format(axis))]
    params += [StringParameter(qualifier='{}label'.format(axis), value=kwargs.get('{}label'.format(axis), '<auto>'), description='Label for the {}-axis (or <auto> to base on {} and {}unit)'.format(axis, axis, axis))]
    params += [FloatArrayParameter(qualifier='{}lim'.format(axis), value=kwargs.get('{}lim'.format(axis), [np.nan, np.nan]), default_unit=default_unit, description='Limit for the {}-axis'.format(axis))]

    return params

def lc(**kwargs):
    params = []

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'times'), choices=['times'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'fluxes'), choices=['fluxes'], description='Array to plot along y-axis')]

    params += _units_labels_lims('x', default_unit=u.d, **kwargs)
    params += _units_labels_lims('y', default_unit=u.W/u.m**2, **kwargs)

    return ParameterSet(params)


def rv(**kwargs):
    params = []

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'times'), choices=['times'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'rvs'), choices=['rvs'], description='Array to plot along y-axis')]

    params += _units_labels_lims('x', default_unit=u.d, **kwargs)
    params += _units_labels_lims('y', default_unit=u.km/u.s,  **kwargs)

    return ParameterSet(params)


def etv(**kwargs):
    raise NotImplementedError


def ifm(**kwargs):
    raise NotImplementedError

def orb(**kwargs):
    raise NotImplementedError


def mesh(**kwargs):
    raise NotImplementedError
