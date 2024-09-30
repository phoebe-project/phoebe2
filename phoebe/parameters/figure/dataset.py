from phoebe.parameters import *
from phoebe.parameters.unit_choices import unit_choices as _unit_choices
import numpy as np

from .common import _use_autofig, _mplcmaps, _mplcolors, _mplmarkers, _mpllinestyles, MPLPropCycler, _label_units_lims, _figure_style_sources, _figure_style_nosources, _figure_uncover_highlight_animate


def lc(b, **kwargs):
    params = []

    # TODO: set hierarchy needs to update choices of any x,y,z in context='figure' to include phases:* a la pblum_ref

    params += [SelectParameter(qualifier='contexts', value=kwargs.get('contexts', '*'), choices=['dataset', 'model'], description='Contexts to include in the plot')]
    params += [SelectParameter(qualifier='datasets', value=kwargs.get('datasets', '*'), choices=[''], description='Datasets to include in the plot')]
    params += [SelectParameter(qualifier='models', value=kwargs.get('models', '*'), choices=[''], description='Models to include in the plot')]

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'times'), choices=['times', 'phases'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='period', visible_if='[context][figure][kind]dperdt:!0.0,x:phases', value=kwargs.get('period', 'period'), choices=['period', 'period_anom'], advanced=True, description='Period to use when phasing for x-axis')]
    params += [ChoiceParameter(qualifier='dpdt', visible_if='[context][figure][kind]dpdt:!0.0,x:phases', value=kwargs.get('dpdt', 'dpdt'), choices=['dpdt', 'none'], advanced=True, description='dpdt to use when phasing for x-axis')]
    params += [ChoiceParameter(qualifier='t0', visible_if='x:phases', value=kwargs.get('t0', 't0_supconj'), choices=['t0_supconj', 't0_perpass', 't0_ref'], advanced=True, description='t0 to use when phasing for x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'fluxes'), choices=['fluxes', 'residuals'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', visible_if='x:times', default_unit=u.d, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:!times', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)

    params += _label_units_lims('y', default_unit=u.W/u.m**2, is_default=True, **kwargs)

    kwargs.setdefault('color', 'black' if _use_autofig else None)
    kwargs.setdefault('marker', '.' if _use_autofig else None)
    kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    params += _figure_style_sources(b, default_color='model', default_marker='manual', default_linestyle='manual', **kwargs)

    params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    params += _figure_uncover_highlight_animate(b, **kwargs)

    return ParameterSet(params)


def rv(b, **kwargs):
    params = []

    params += [SelectParameter(qualifier='contexts', value=kwargs.get('contexts', '*'), choices=['dataset', 'model'], description='Contexts to include in the plot')]
    params += [SelectParameter(qualifier='datasets', value=kwargs.get('datasets', '*'), choices=[''], description='Datasets to include in the plot')]
    params += [SelectParameter(qualifier='models', value=kwargs.get('models', '*'), choices=[''], description='Models to include in the plot')]
    params += [SelectParameter(qualifier='components', value=kwargs.get('components', '*'), choices=[''], description='Components to include in the plot')]

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'times'), choices=['times', 'phases'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='period', visible_if='[context][figure][kind]dperdt:!0.0,x:phases', value=kwargs.get('period', 'period'), choices=['period', 'period_anom'], advanced=True, description='Period to use when phasing for x-axis')]
    params += [ChoiceParameter(qualifier='dpdt', visible_if='[context][figure][kind]dpdt:!0.0,x:phases', value=kwargs.get('dpdt', 'dpdt'), choices=['dpdt', 'none'], advanced=True, description='dpdt to use when phasing for x-axis')]
    params += [ChoiceParameter(qualifier='t0', visible_if='x:phases', value=kwargs.get('t0', 't0_supconj'), choices=['t0_supconj', 't0_perpass', 't0_ref'], advanced=True, description='t0 to use when phasing for x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'rvs'), choices=['rvs', 'residuals'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', visible_if='x:times', default_unit=u.d, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:!times', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)

    params += _label_units_lims('y', default_unit=u.km/u.s, is_default=True, **kwargs)

    kwargs.setdefault('color', 'black') if _use_autofig else None
    kwargs.setdefault('marker', '.' if _use_autofig else None)
    kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    params += _figure_style_sources(b, default_color='component', default_marker='manual', default_linestyle='manual', **kwargs)

    params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    params += _figure_uncover_highlight_animate(b, **kwargs)

    return ParameterSet(params)


# def etv(b, **kwargs):
#     params = []
#
#     params += [SelectParameter(qualifier='dataset', value=kwargs.get('dataset', '*'), choices=[''], description='Datasets to include in the plot')]
#     params += [SelectParameter(qualifier='model', value=kwargs.get('model', '*'), choices=[''], description='Models to include in the plot')]
#
#     params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'time_ephems'), choices=['time_ephems', 'Ns'], description='Array to plot along x-axis')]
#     params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'etvs'), choices=['etvs', 'time_ecls'], description='Array to plot along y-axis')]
#
#     params += _label_units_lims('x', visible_if='x:time_ephems', default_unit=u.d, is_default=True, **kwargs)
#     params += _label_units_lims('x', visible_if='x:Ns', default_unit=u.dimensionless_unscaled, **kwargs)
#
#     params += _label_units_lims('y', visible_if='y:etvs', default_unit=u.d, is_default=True, **kwargs)
#     params += _label_units_lims('y', visible_if='y:time_ecls', default_unit=u.d, **kwargs)
#
#     return ParameterSet(params)


def orb(b, **kwargs):
    params = []

    params += [SelectParameter(qualifier='datasets', value=kwargs.get('datasets', '*'), choices=[''], description='Datasets to include in the plot')]
    params += [SelectParameter(qualifier='models', value=kwargs.get('models', '*'), choices=[''], description='Models to include in the plot')]
    params += [SelectParameter(qualifier='components', value=kwargs.get('components', '*'), choices=[''], description='Components to include in the plot')]

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'us'), choices=['times', 'phases', 'us', 'vs', 'ws', 'vus', 'vvs', 'vws'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='period', visible_if='[context][figure][kind]dperdt:!0.0,x:phases', value=kwargs.get('period', 'period'), choices=['period', 'period_anom'], advanced=True, description='Period to use when phasing for x-axis')]
    params += [ChoiceParameter(qualifier='dpdt', visible_if='[context][figure][kind]dpdt:!0.0,x:phases', value=kwargs.get('dpdt', 'dpdt'), choices=['dpdt', 'none'], advanced=True, description='dpdt to use when phasing for x-axis')]
    params += [ChoiceParameter(qualifier='t0', visible_if='x:phases', value=kwargs.get('t0', 't0_supconj'), choices=['t0_supconj', 't0_perpass', 't0_ref'], advanced=True, description='t0 to use when phasing for x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'ws'), choices=['us', 'vs', 'ws', 'vus', 'vvs', 'vws'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', visible_if='x:times', default_unit=u.d, is_default=False, **kwargs)
    # TODO: this visible_if will likely fail if/once we implement phases:* for the choices for x
    params += _label_units_lims('x', visible_if='x:phases', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)
    params += _label_units_lims('x', visible_if='x:us|vs|ws', default_unit=u.solRad, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:vus|vvs|vws', default_unit=u.km/u.s, is_default=False, **kwargs)

    params += _label_units_lims('y', visible_if='y:us|vs|ws', default_unit=u.solRad, is_default=True, **kwargs)
    params += _label_units_lims('y', visible_if='y:vus|vvs|vws', default_unit=u.km/u.s, is_default=False, **kwargs)

    kwargs.setdefault('color', 'black' if _use_autofig else None)
    kwargs.setdefault('marker', '.' if _use_autofig else None)
    kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    params += _figure_style_sources(b, default_color='component', default_marker='manual', default_linestyle='manual', **kwargs)

    params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    params += _figure_uncover_highlight_animate(b, **kwargs)

    return ParameterSet(params)

def lp(b, **kwargs):
    params = []

    params += [SelectParameter(qualifier='contexts', value=kwargs.get('contexts', '*'), choices=['dataset', 'model'], description='Contexts to include in the plot')]
    params += [SelectParameter(qualifier='datasets', value=kwargs.get('datasets', '*'), choices=[''], description='Datasets to include in the plot')]
    params += [SelectParameter(qualifier='models', value=kwargs.get('models', '*'), choices=[''], description='Models to include in the plot')]
    # params += [SelectParameter(qualifier='times', value=kwargs.get('times', '*'), choices=[''], description='Times to include in the plot')]
    params += [SelectParameter(qualifier='components', value=kwargs.get('components', '*'), choices=[''], description='Components to include in the plot')]

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'wavelengths'), choices=['wavelengths'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'flux_densities'), choices=['flux_densities'], description='Array to plot along y-axis')]

    params += _label_units_lims('x', default_unit=u.nm, is_default=True, **kwargs)
    params += _label_units_lims('y', default_unit=u.W/(u.m**2*u.nm), is_default=True, **kwargs)

    kwargs.setdefault('color', 'black' if _use_autofig else None)
    kwargs.setdefault('marker', '.' if _use_autofig else None)
    kwargs.setdefault('linestyle', 'solid' if _use_autofig else None)
    params += _figure_style_sources(b, default_color='component', default_marker='manual', default_linestyle='manual', **kwargs)

    params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', True), advanced=True, description='Whether to draw the legend')]

    params += _figure_uncover_highlight_animate(b, **kwargs)

    return ParameterSet(params)

def mesh(b, **kwargs):
    params = []

    params += [SelectParameter(qualifier='datasets', value=kwargs.get('datasets', '*'), choices=[''], description='Datasets to include in the plot')]
    params += [SelectParameter(qualifier='models', value=kwargs.get('models', '*'), choices=[''], description='Models to include in the plot')]
    # params += [SelectParameter(qualifier='times', value=kwargs.get('times', '*'), choices=[''], description='Times to include in the plot')]
    params += [SelectParameter(qualifier='components', value=kwargs.get('components', '*'), choices=[''], description='Components to include in the plot')]

    params += [ChoiceParameter(qualifier='x', value=kwargs.get('x', 'us'), choices=['us', 'vs', 'ws', 'xs', 'ys', 'zs'], description='Array to plot along x-axis')]
    params += [ChoiceParameter(qualifier='y', value=kwargs.get('y', 'vs'), choices=['us', 'vs', 'ws', 'xs', 'ys', 'zs'], description='Array to plot along y-axis')]


    params += _label_units_lims('x', visible_if='x:us|vs|ws', default_unit=u.solRad, is_default=True, **kwargs)
    params += _label_units_lims('x', visible_if='x:xs|ys|zs', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)

    params += _label_units_lims('y', visible_if='y:us|vs|ws', default_unit=u.solRad, is_default=True, **kwargs)
    params += _label_units_lims('y', visible_if='y:xs|ys|zs', default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)

    params += [ChoiceParameter(qualifier='fc_source', value=kwargs.get('fc_source', 'column'), choices=['column', 'manual', 'component', 'model'], description='Source to use for facecolor.  For column, see the fc_column parameter.  For manual, see the fc parameter.  Otherwise, see the color parameter tagged with the corresponding component/model')]
    params += [ChoiceParameter(qualifier='fc_column', visible_if='fc_source:column', value=kwargs.get('fc_column', 'None'), choices=['None'], description='Column from the mesh to plot as facecolor if fc_source is column.')]
    params += [ChoiceParameter(qualifier='fc', visible_if='fc_source:manual', value=kwargs.get('fc', 'None'), choices=['None']+_mplcolors, description='Color to use as facecolor if fc_source is manual.')]
    params += [ChoiceParameter(qualifier='fcmap_source', visible_if='fc_source:column', value=kwargs.get('fcmap_source', 'auto'), choices=['auto', 'manual'], description='Source to use for facecolor colormap.  To have fcmap adjust based on the value of fc, use auto.  For manual, see the fcmap parameter.')]
    params += [ChoiceParameter(qualifier='fcmap', visible_if='fc_source:column,fcmap_source:manual', value=kwargs.get('fcmap', 'viridis'), choices=_mplcmaps, description='Colormap to use for facecolor if fcmap_source=\'manual\'.')]

    params += [ChoiceParameter(qualifier='ec_source', value=kwargs.get('ec_source', 'manual'), choices=['column', 'manual', 'component', 'model', 'face'], description='Source to use for edgecolor.  For column, see the ec_column parameter.  For manual, see the ec parameter.  Otherwise, see the color parameter tagged with the corresponding component/model')]
    params += [ChoiceParameter(qualifier='ec_column', visible_if='ec_source:column', value=kwargs.get('ec_column', 'None'), choices=['None'], description='Column from the mesh to plot as edgecolor if ec_source is column.')]
    params += [ChoiceParameter(qualifier='ec', visible_if='ec_source:manual', value=kwargs.get('ec', 'black' if _use_autofig else None), choices=['face']+_mplcolors, description='Color to use as edgecolor if ec_source is manual.')]
    params += [ChoiceParameter(qualifier='ecmap_source', visible_if='ec_source:column', value=kwargs.get('ecmap_source', 'auto'), choices=['auto', 'manual'], description='Source to use for edgecolor colormap.  To have ecmap adjust based on the value of ec, use auto.  For manual, see the ecmap parameter.')]
    params += [ChoiceParameter(qualifier='ecmap', visible_if='ec_source:column,ecmap_source:manual', value=kwargs.get('fcmap', 'viridis'), choices=_mplcmaps, description='Colormap to use for edgecolor if ecmap_source=\'manual\'.')]




    for q in ['fc', 'ec']:
        # see dataset._mesh_columns
        # even though this isn't really the default case, we'll pass is_default=True
        # so that we get fc/eclim_source, etc, parameters
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:vxs|vys|vzs|vus|vws|vvs|rvs'.format(q,q), default_unit=u.km/u.s, is_default=True, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:rs|rprojs'.format(q,q), default_unit=u.solRad, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:areas'.format(q,q), default_unit=u.solRad**2, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:teffs'.format(q,q), default_unit=u.K, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:pblum_ext|abs_pblum_ext'.format(q,q), default_unit=u.W, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:ptfarea'.format(q,q), default_unit=u.m, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:intensities|normal_intensities|abs_intensities|abs_normal_intensities'.format(q,q), default_unit=u.W/u.m**3, is_default=False, **kwargs)
        params += _label_units_lims(q, visible_if='{}_source:column,{}_column:visibilities|mus|loggs|boost_factors|ldint'.format(q,q), default_unit=u.dimensionless_unscaled, is_default=False, **kwargs)

    # TODO: legend=True currently fails
    params += [BoolParameter(qualifier='draw_sidebars', value=kwargs.get('draw_sidebars', True), advanced=True, description='Whether to draw the sidebars')]
    params += [BoolParameter(qualifier='legend', value=kwargs.get('legend', False), advanced=True, description='Whether to draw the legend')]

    params += _figure_uncover_highlight_animate(b, uncover=False, highlight=False, **kwargs)

    return ParameterSet(params)
