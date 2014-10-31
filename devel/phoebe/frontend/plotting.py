from phoebe.parameters import parameters
from phoebe.units import conversions
from phoebe.utils import plotlib
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from scipy import interpolate
import logging

logger = logging.getLogger("FRONT.PLOTTING")
logger.addHandler(logging.NullHandler())


def _defaults_from_dataset(b, ds, kwargs):
    """
    provide defaults for any parameters not provided in kwargs (which should include args)
    """
    ref = ds['ref']
    context = ds.context
    category = context[:-3]
    typ = context[-3:]
    
    kwargs.setdefault('label', '{} ({})'.format(ref, typ))
    
    #~ kwargs.setdefault('repeat', 0)
    # Phases are default only when obs are given in phase
    default_phased = not 'time' in ds['columns'] and 'phase' in ds['columns']
    kwargs.setdefault('phased', b.get_system().get_label() if default_phased else 'False')
    
    xk, yk, xl, yl = _xy_from_category(category)
    xl = 'Phase' if kwargs['phased'] != 'False' else xl
    #x, y = ds[xk], ds[yk]

    kwargs.setdefault('xunit', 'cy' if kwargs['phased'] != 'False' else ds.get_parameter(xk).get_unit())
    kwargs.setdefault('yunit', ds.get_parameter(yk).get_unit())

    try:
        xunit = conversions.unit2texlabel(kwargs['xunit'])
    except:
        xunit = kwargs['xunit']
    try:
        yunit = conversions.unit2texlabel(kwargs['yunit'])
    except:
        yunit = kwargs['yunit']
    
    # If we have more than one period in the system and we're phased, 
    # then let's adjust the xlabel to also tell which period is being used
    if kwargs['phased'] != 'False' and len(b.twigs('period'))>1:
        xl = '{} of {}'.format(xl, kwargs['phased'])
        
    kwargs.setdefault('xlabel', '{} ({})'.format(xl, xunit))
    kwargs.setdefault('ylabel', '{} ({})'.format(yl, yunit))

    kwargs.setdefault('xlim', '_auto_')
    kwargs.setdefault('ylim', '_auto_')

    # Reverse axes when plotting in magnitude
    if yunit is not None and 'mag' in yunit and not simulate:
        ylim = ax.get_ylim()
        if ylim[0] < ylim[1]:
            ax.set_ylim(ylim[::-1]) 
            
    return kwargs

def _from_dataset(b, twig, context):
    """
    helper function to retrieve a dataset from the bundle
    """
    dsti = b._get_by_search(twig, context=context, class_name='*DataSet',
                                return_trunk_item=True, all=True)
                                
    kwargs_defaults = {}
                                
    # It's possible that we need to plot an SED: in that case, we have
    # more than one dataset but they are all light curves that are grouped
    if len(dsti) > 1:
        # retrieve obs for sed grouping.
        dstiobs = b._get_by_search(twig, context='*obs', class_name='*DataSet',
                               return_trunk_item=True, all=True)
        
        # Check if they are all lightcurves and collect group name
        groups = []
        if not len(dsti) == len(dstiobs):
            raise ValueError("Cannot plot synthetics of SED '{}'".format(twig))
        
        for idsti, jdsti in zip(dsti, dstiobs):
            correct_category = idsti['item'].get_context()[:-3] == 'lc'
            is_grouped = 'group' in jdsti['item']                
            if not correct_category or not is_grouped:
                # raise the correct error:
                b._get_by_search(twig, context='*syn', class_name='*DataSet',
                               return_trunk_item=True)
            else:
                groups.append(jdsti['item']['group'])
        # check if everything belongs to the same group
        if not len(set(groups)) == 1:
            raise KeyError("More than one SED group found matching twig '{}', please be more specific".format(twig))
        
        obj = b.get_object(dsti[0]['label'])
        context = 'sed' + dsti[0]['item'].get_context()[-3:]
        ds = dict(ref=groups[0])
    else:
        dsti = dsti[0]
        ds = dsti['item']
        obj = b.get_object(dsti['label'])
        context = ds.get_context()
        
    return ds, context, kwargs_defaults 

def _xy_from_category(category):
    """
    returns the x and y arrays given a dataset and its category
    """
    
    # TODO: handle phase here
    if category=='lc':
        xk = 'time'
        yk = 'flux'
        xl = 'Time'
        yl = 'Flux'
    elif category=='rv':
        xk = 'time'
        yk = 'rv'
        xl = 'Time'
        yl = 'RV'
    elif category=='sp':
        # TODO: check these
        xk = 'wavelength'
        yk = 'flux'
        xl = 'Wavelength'
        yl = 'Flux'
    elif category=='etv':
        xk = 'time'
        yk = 'etv'
        xl = 'Time'
        yl = 'ETV'
    else:
        logger.warning("{} category not currently supported in frontend plotting".format(category))
        xk = None
        yk = None
        xl = None
        yl = None
        
    return xk, yk, xl, yl 

def _kwargs_defaults_override(kwargs_defaults, kwargs):
    """
    adjust kwargs_defaults so that any values in kwargs trump
    """
    for k,v in kwargs.items():
        if k in kwargs_defaults and isinstance(kwargs_defaults[k], dict):
            kwargs_defaults[k]['value'] = v
        else:
            kwargs_defaults[k] = v
    return kwargs_defaults

def _plot(b, t, ds, context, kwargs_defaults, **kwargs):
    """
    [FUTURE]
    shared function for most dataset plotting (obs, syn, etc)
    """
    # let's make sure when we send kwargs_defaults to build labels, etc, that
    # we include user-sent units, etc
    kwargs_defaults = _kwargs_defaults_override(kwargs_defaults, kwargs)
    kwargs_defaults = _defaults_from_dataset(b, ds, kwargs_defaults)
    category = context[:-3]
    typ = context[-3:]
    
    
    kwargs_defaults['fmt'] = {'value': 'k-' if typ=='syn' else 'k.', 'description': 'matplotlib format'}
    kwargs_defaults['uncover'] = {'value': False, 'description': 'only show up to the current time (must be passed during draw call)', 'cast_type': 'bool'}
    kwargs_defaults['highlight'] = {'value': typ=='syn', 'description': 'draw a marker at the current time (must be passed during draw call)', 'cast_type': 'bool'}
    kwargs_defaults['highlight_fmt'] = {'value': 'ko', 'description': 'matplotlib format for time if higlight is True'}
    kwargs_defaults['highlight_ms'] = {'value': 5, 'description': 'matplotlib markersize for time if highlight is True', 'cast_type': 'int'}
    kwargs_defaults['scroll'] = {'ps': 'axes', 'value': False, 'description': 'whether to override xlim and scroll when time is passed during draw call', 'cast_type': 'make_bool'}
    kwargs_defaults['scroll_xlim'] = {'ps': 'axes', 'value': [-2,2], 'description': 'the xlims to provide relative to the current time if scroll==True and time is passed during draw call', 'cast_type': 'list'}

    xk, yk, xl, yl = _xy_from_category(category)  # TODO: we also call this in _defaults_from_dataset, let's consolidate
    kwargs_defaults = _kwargs_defaults_override(kwargs_defaults, kwargs)    
    mpl_kwargs = {k:v['value'] if isinstance(v,dict) else v for k,v in kwargs_defaults.items() if v not in ['_auto_']}

    phased = kwargs_defaults.get('phased', 'False')
    
    if typ=='obs' or ds.enabled:
        x, y = ds.get_value(xk, 'd' if phased != 'False' else kwargs_defaults['xunit']), ds.get_value(yk, kwargs_defaults['yunit'])
        
        if category=='sp':
            # then we need the index for the current time
            t_ind = np.where(ds.get_value('time')==t)
            if len(t_ind):
                x = x[t_ind[0]]
                y = y[t_ind[0]]
                if len(x):
                    x = x[0]
                    y = y[0]
            else:
                x, y = np.array([]), np.array([])
            
    else:
        x, y = np.array([]), np.array([])

    if xk=='time' and phased != 'False':
        # the phase array probably isn't filled, so we need to compute phase now

        ephem = b.get_ephem(phased)
        period = ephem.get('period', 1.0)
        t0 = ephem.get('t0', 0.0)
        
        # TODO: allow passing different orbital levels (maybe take twig to orbit or to period and t0?)
        x = ((x-t0) % period) / period
        
        # sort phases
        sa = np.argsort(x)
        x, y = x[sa], y[sa]

    if t and xk=='time' and len(x) and phased=='False' and kwargs_defaults['uncover']['value']:
        xd = x[x<=t]
        yd = y[x<=t]
    else:
        xd = x
        yd = y

    if kwargs_defaults['highlight']['value']:
        mpl_select_kwargs = {k.split('_')[1]:v['value'] if isinstance(v,dict) else v for k,v in kwargs_defaults.items() if k.split('_')[0]=='highlight' and k!='highlight'}
        
        if t and xk=='time' and len(x) and phased=='False' and t>min(x) and t<max(x):
            interp = interpolate.interp1d(x, y, bounds_error=False)
            s_x = [t]
            s_y = [interp(t)]
        else:
            s_x, s_y = [], []
        
        return ['plot' if typ=='syn' else 'errorbar', 'plot'], [(xd, yd), (s_x, s_y)], [mpl_kwargs, mpl_select_kwargs], kwargs_defaults
        
    else:
        return 'plot' if typ=='syn' else 'errorbar', (xd, yd), mpl_kwargs, kwargs_defaults

def obs(b, t, datatwig, **kwargs):
    """
    This is a preprocessing function for :py:func:`Bundle.attach_plot`
    
    plot an observation dataset given its datatwig
    """
    
    ds, context, kwargs_defaults = _from_dataset(b, datatwig, '*obs')
    return _plot(b, t, ds, context, kwargs_defaults, **kwargs)
    
def syn(b, t, datatwig, **kwargs):
    """
    [FUTURE]
    This is a preprocessing function for :py:func:`Bundle.attach_plot`
    
    plot a synthetic dataset given its twig
    """
    ds, context, kwargs_defaults = _from_dataset(b, datatwig, '*syn')
    return _plot(b, t, ds, context, kwargs_defaults, **kwargs)
    
def residuals(b, t, datatwig, **kwargs):
    """
    [FUTURE]
    This is a preprocessing function for :py:func:`Bundle.attach_plot`

    """
    obs_ds, obs_context, kwargs_defaults = _from_dataset(b, datatwig, '*obs')
    syn_ds, syn_context, dump = _from_dataset(b, datatwig, '*syn')
    # TODO: support using _plot somehow
    
    kwargs_defaults = _defaults_from_dataset(b, ds, kwargs_defaults)

    kwargs_defaults = _kwargs_defaults_override(kwargs_defaults, kwargs)

    return 'errorbar', (x, y), {}, kwargs_defaults
    
def mesh(b, t, objref, dataref, **kwargs):
    """
    [FUTURE]
    This is a preprocessing function for :py:func:`Bundle.attach_plot`
    
    """
    
    # unpack expected kwargs
    select = kwargs.get('select', 'teff')
    cmap = kwargs.get('cmap', None)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    
    kwargs_defaults = {}
        
    # TODO: make these options:
    boosting_alg = 'none'
    with_partial_as_half = True
    antialiasing = True
    

    obj = b.get_object(objref)
    if t:
        obj.set_time(t)
    try:
        total_flux = obj.projected_intensity(ref=dataref,boosting_alg=boosting_alg,
                                      with_partial_as_half=with_partial_as_half)
    except ValueError as msg:
        raise ValueError(str(msg)+'\nPossible solution: did you set the time (set_time) of the system?')
    except AttributeError as msg:
        total_flux = 0.0
        logger.warning("Body has not attribute `projected_intensity', some stuff will not work")
    
    mesh = b.get_mesh(objref)
    
    # in case time has never been set:
    if not len(mesh):
        b.set_time(0)
        mesh = b.get_mesh(objref)
    
    # Order the body's triangles from back to front so that they get plotted in
    # the right order.
    sa = np.argsort(mesh['center'][:, 2])
    mesh = mesh[sa]

    x, y = mesh['center'][:, 0],mesh['center'][:, 1]
    
    # Default color maps and background depend on the type of dependables:
    kwargs_defaults['background'] = 'k'
    if cmap is None and select == 'rv':
        cmap = pl.cm.RdBu_r
    elif cmap is None and select == 'teff':
        cmap = pl.cm.afmhot
        kwargs_defaults['background'] = 0.7
    elif cmap is None and select == 'logg':
        cmap = pl.cm.gnuplot
        kwargs_defaults['background'] = 0.7
    elif cmap is None and select[0] == 'B':
        cmap = pl.cm.jet
    elif cmap is None:
        cmap = pl.cm.gray

    # Default lower and upper limits for colors:
    vmin_ = vmin
    vmax_ = vmax
    
    # Set the values and colors of the triangles: there's a lot of possibilities
    # here: we can plot the projected intensity (i.e. as we would see it), but
    # we can also plot other quantities like the effective temperature, radial
    # velocity etc...
    cmap_ = None
    norm_proj = None
    if select == 'proj':
        colors = np.where(mesh['mu']>0, mesh['proj_'+dataref] / mesh['mu'],0.0)       
        #if 'refl_'+ref in mesh.dtype.names:
        #    colors += mesh['refl_'+ref]
        norm_proj = colors.max()
        colors /= norm_proj
        values = colors
        if vmin is None:
            vmin_ = 0
        if vmax is None:
            vmax_ = 1
        
    else:
        if select == 'rv':
            values = -mesh['velo___bol_'][:, 2] * 8.049861
        elif select == 'intensity':
            values = mesh['ld_'+ref+'_'][:, -1]
        elif select == 'proj2':
            values = mesh['proj_'+ref] / mesh['mu']
            if 'refl_'+ref in mesh.dtype.names:
                values += mesh['refl_'+ref]
        elif select == 'Bx':
            values = mesh['B_'][:, 0]
        elif select == 'By':
            values = mesh['B_'][:, 1]
        elif select == 'Bz':
            values = mesh['B_'][:, 2]    
        elif select == 'B':
            values = np.sqrt(mesh['B_'][:, 0]**2 + \
                             mesh['B_'][:, 1]**2 + \
                             mesh['B_'][:, 2]**2)
        elif select in mesh.dtype.names:
            values = mesh[select]
        else:
            values = select[sa]
        # Set the limits of the color scale, if we need to compute them
        # ourselves
        vmin, vmax = None, None # TODO: remove when add vmin, vmax as args/kwargs
        if vmin is None:
            vmin_ = values[mesh['mu'] > 0].min()
        if vmax is None:
            vmax_ = values[mesh['mu'] > 0].max()
        
        # Special treatment for black body map, since the limits need to be
        # fixed for the colors to match the temperature
        if len(values.shape)==1:
            colors = (values - vmin_) / (vmax_ - vmin_)
        else:
            colors = values
        if cmap == 'blackbody' or cmap == 'blackbody_proj' or cmap == 'eye':
            cmap_ = cmap
            cmap = plotlib.blackbody_cmap()
            vmin_, vmax_ = 2000, 20000
            colors = (values-vmin_) / (vmax_-vmin_)
        # Limbdarkened velocity map
        if select == 'rv' and cmap == 'proj':
            proj = mesh['proj_'+ref]
            proj = proj / proj.max()
            rvmax = values.max()
            offset = -rvmax
            values = values - offset
            scale = 2*rvmax
            values = values/scale
            colors = pl.cm.RdBu_r(values)*proj[:,None]
            values = values.reshape((-1,1))
            colors[:,3] = 1.0
            if vmin is None:
                vmin_ = offset
            if vmax is None:
                vmax_ = -offset
                
    mpl_args = (mesh['triangle'].reshape((-1, 3, 3))[:, :, :2], None, False)
    #~ mpl_kwargs = {'array': values}
   
    mpl_kwargs = {'array': values,
                'antialiaseds': antialiasing,
                'edgecolors': cmap(colors),
                'facecolors': cmap(colors),
                'cmap': cmap}
                
    if not cmap_ in ['blackbody_proj', 'eye'] and len(values.shape)==1:
        mpl_kwargs = {'array': values,
                    'antialiaseds': antialiasing,
                    'edgecolors': cmap(colors),
                    'facecolors': cmap(colors),
                    'cmap': cmap}
        
        
    
    # When a RGB select color values is given
    elif not cmap_ in ['blackbody_proj', 'eye']:
        mpl_kwargs = {'edgecolors': colors,
                      'antialiaseds': antialiasing,
                      'facecolors': colors}

    elif cmap_ == 'blackbody_proj':
        # In this particular case we also need to set the values for the
        # triangles first
        values = np.abs(mesh['proj_'+dataref] / mesh['mu'])
        if 'refl_'+dataref in mesh.dtype.names:
            values += mesh['refl_'+dataref]
        scale = vmax if vmax is not None else 1.0
        values = (values / (scale*values.max())).reshape((-1, 1)) * np.ones((len(values), 4))
        values[:, -1] = 1.0
        colors = np.array([cmap(c) for c in colors]) * values 

        mpl_kwargs = {'antialiaseds': antialiasing,
                      'edgecolors': colors,
                      'facecolors': colors}

        
    elif cmap_ == 'eye':
        values = np.abs(mesh['proj_'+dataref] / mesh['mu'])
        if 'refl_'+dataref in mesh.dtype.names:
            values += mesh['refl_'+dataref]
        keep = values > (0.5*values.max())
        values = values / values[keep].min()
        values = values.reshape((-1, 1)) * np.ones((len(values), 4))
        values[:, -1] = 1.0
        colors = np.array([cmap(c) for c in colors]) * values
        colors[colors > 1] = 1.0
        
        mpl_kwargs = {'antialiaseds': antialiasing,
                      'edgecolors': colors,
                      'facecolors': colors}

    # Set the color scale limits
    #~ if vmin is not None: vmin_ = vmin
    #~ if vmax is not None: vmax_ = vmax
    #~ p.set_clim(vmin=vmin_,vmax=vmax_)
    
    # Derive the limits for the axis
    # (dont be smart when axis where given)
    offset_x = (x.min() + x.max()) / 2.0
    offset_y = (y.min() + y.max()) / 2.0
    margin = 0.01 * x.ptp()
    lim_max = max(x.max() - x.min(), y.max() - y.min())
    
    xlim = offset_x - margin - lim_max/2.0,offset_x + lim_max/2.0 + margin
    ylim = offset_y - margin - lim_max/2.0,offset_y + lim_max/2.0 + margin
    
    kwargs_defaults['xlim'] = xlim
    kwargs_defaults['ylim'] = ylim
    
    #~ kwargs_defaults['xlabel'] = 'X Position (R_solar)'
    #~ kwargs_defaults['ylabel'] = 'Y Position (R_solar)'
    
    _kwargs_defaults_override(kwargs_defaults, kwargs)
    return 'PolyCollection', mpl_args, mpl_kwargs, kwargs_defaults
    
def orbit(b, t, objref, dataref, x='x', y='y', **kwargs):
    """
    [FUTURE]
    This is a preprocessing function for :py:func:`Bundle.attach_plot`
    
    """
    ds, context = _from_dataset(b, '{}@{}'.format(dataref, objref), 'orbsyn')  # TODO: there should be a better way to do this, especially if dataref@objref is passed as twig
    
    times = ds['bary_time'] # or ds['prop_time']
    pos = ds['position']
    vel = ds['velocity']
    
    positions = ['x','y','z']
    velocities = ['vx','vy','vz']
    if x in positions:
        xdata = pos[positions.index(x)]
    elif x in velocities:
        xdata = vel[positions.index(x)]
    else: # time
        xdata = times
    if y in positions:
        ydata = pos[positions.index(y)]
    elif y in velocities:
        ydata = vel[positions.index(y)]
    else: # time
        ydata = times
        
    kwargs_defaults = {}
    kwargs_defaults['fmt'] = {'value': 'k-' if typ=='syn' else 'k.', 'description': 'matplotlib format'}
    kwargs_defaults['highlight'] = {'value': typ=='syn', 'description': 'draw a marker at the current time (must be passed during draw call)', 'cast_type': 'bool'}
    kwargs_defaults['highlight_fmt'] = {'value': 'ko', 'description': 'matplotlib format for time if higlight is True'}
    kwargs_defaults['highlight_ms'] = {'value': 5, 'description': 'matplotlib markersize for time if highlight is True', 'cast_type': 'int'}
    #~ kwargs_defaults['scroll'] = {'ps': 'axes', 'value': False, 'description': 'whether to override xlim and scroll when time is passed during draw call', 'cast_type': 'make_bool'}
    #~ kwargs_defaults['scroll_xlim'] = {'ps': 'axes', 'value': [-2,2], 'description': 'the xlims to provide relative to the current time if scroll==True and time is passed during draw call', 'cast_type': 'list'}
    #~ kwargs_defaults['scroll_ylim'] = {'ps': 'axes', 'value': [-2,2], 'description': 'the ylims to provide relative to the current time if scroll==True and time is passed during draw call', 'cast_type': 'list'}

    
    # TODO: aspect ratio
    
    kwargs_defaults = _kwargs_defaults_override(kwargs_defaults, kwargs)
    mplkwargs = {k:v['value'] if isinstance(v,dict) else v for k,v in kwargs_defaults.items() if v not in ['_auto_']}
    
    if kwargs_defaults['highlight']['value']:
        mpl_select_kwargs = {k.split('_')[1]:v['value'] if isinstance(v,dict) else v for k,v in kwargs_defaults.items() if k.split('_')[0]=='highlight' and k!='highlight'}
        
        if t and len(xdata) and phased=='False' and t>min(times) and t<max(times):
            interp_x = interpolate.interp2d(times, xdata, bounds_error=False)
            interp_y = interpolate.interp2d(times, ydata, bounds_error=False)
            #~ s_t = [t]
            s_x = [interp(t)]
            s_y = [interp(t)]
        else:
            s_x, s_y = [], []
        
        return ['plot', 'plot'], [(xdata, ydata), (s_x, s_y)], [mplkwargs, mpl_select_kwargs], kwargs_defaults
        
    else:
        return 'plot', (xdata, ydata), mplkwargs, kwargs_defaults
    
    
def xy(b, t, x, y, **kwargs):
    """
    [FUTURE]
    This is a preprocessing function for :py:func:`Bundle.attach_plot`
    
    """

    # x and y here are twigs to the x and y data
    # we want to guess the context from the y data and apply defaults
    
    # let's get context from y
    context = b._get_by_search(y, return_trunk_item=True)['context']
    typ = context[-3:]
    y_ds = b._get_by_search('@'.join(y.split('@')[1:]))
    kwargs_defaults = _defaults_from_dataset(b, y_ds , kwargs)
    
    if typ=='obs':
        mplcmd = 'errorbar'
        kwargs_defaults['fmt'] = {'value': 'k.', 'description': 'matplotlib format'}
    elif typ=='syn':
        mplcmd = 'plot'
        kwargs_defaults['fmt'] = {'value': 'k-', 'description': 'matplotlib format'}
    else:
        raise NotImplementedError
    
    #~ category = context[:-3]
    
    # TODO: handle retrieving phase correctly (may not be filled)
    xdata = b.get(x)   # TODO: deal with units
    ydata = b.get(y)   # TODO: deal with units
    
    kwargs_defaults = _kwargs_defaults_override(kwargs_defaults, kwargs)
    
    return mplcmd, (xdata, ydata), {}, kwargs_defaults
    
def xyz(b, t, x, y, z, **kwargs):
    """
    [FUTURE]
    This is a preprocessing function for :py:func:`Bundle.attach_plot`
    
    """
    raise NotImplementedError
    
def mplcommand(b, t, *args, **kwargs):
    """
    [FUTURE]
    This is a preprocessing function for :py:func:`Bundle.attach_plot`
    
    kwargs can include the following:
    
    func: (the name of a matplotlib function that
    is an attribute of either ax or matplotlib.collections)
    args: the args to pass to func
    
    and any additional kwargs that will be stored in the plot PS and 
    then passed as kwargs to the plotting call when draw is called
    
    """
    mplfunc = kwargs.get('mplfunc', 'plot')
    mplargs = kwargs.get('mplargs', [])
    if not isinstance(mplargs, list):
        mplargs = list(mplargs)
    
    mplkwargs = {}
    kwargs_defaults = kwargs

    return mplfunc, mplargs, mplkwargs, kwargs_defaults
    
def time_axvline(b, t, *args, **kwargs):
    """
    [FUTURE]
    This is a preprocessing function for :py:func:`Bundle.attach_plot`
    """
    
    kwargs_defaults = {}
    kwargs_defaults['linewidth'] = {'value': 1, 'description': 'matplotlib linewidth', 'cast_type': 'int'}
    kwargs_defaults['linestyle'] = {'value': '-', 'description': 'matplotlib linestyle'}
    kwargs_defaults['color'] = {'value': 'k', 'description': 'matplotlib color'}

    kwargs_defaults = _kwargs_defaults_override(kwargs_defaults, kwargs)
    mplkwargs = {k:v['value'] if isinstance(v,dict) else v for k,v in kwargs_defaults.items() if v not in ['_auto_']}
    return 'axvline', (t,), mplkwargs, kwargs_defaults 

def ds_axvspan(b, t, datatwig, **kwargs):
    """
    [FUTURE]
    This is a preprocessing function for :py:func:`Bundle.attach_plot`
    
    This function creates an axvspan for the time range of the dataset defined
    by datatwig.  This axvspan can be attached to multiple axes and can be
    formatted with different colors depending on whether the current time
    is within the range or not.
    """
    ds, context, kwargs_defaults = _from_dataset(b, datatwig, '*obs')
    xl, xu = min(ds['time']), max(ds['time'])
    
    kwargs_defaults = {}
    kwargs_defaults['color'] = {'value': 'b', 'description': 'matplotlib color'}
    kwargs_defaults['alpha'] = {'value': 0.15, 'description': 'matplotlib alpha (0 to 1)', 'cast_type': 'float'}
    kwargs_defaults['highlight'] = {'value': True, 'description': 'change the format if current time is in range (time must be passed during draw call)', 'cast_type': 'bool'}
    kwargs_defaults['highlight_color'] = {'value': 'b', 'description': 'matplotlib color if highlight is True and time in range'}
    kwargs_defaults['highlight_alpha'] = {'value': 0.5, 'description': 'matplotlib alpha if highlight is True and time in range', 'cast_type': 'float'}
    
    kwargs_defaults = _kwargs_defaults_override(kwargs_defaults, kwargs)
    mplkwargs = {k:v['value'] if isinstance(v,dict) else v for k,v in kwargs_defaults.items() if v not in ['_auto_']}
    
    # now handle changing mplkwargs if we want to apply the highlight
    if kwargs_defaults['highlight'] and t and t <= xu and t >= xl:
        for k,v in mplkwargs.items():
            if k.split('_')[0]=='highlight' and k != 'highlight':
                mplkwargs[k.split('_')[1]] = v
       
    return 'axvspan', (xl, xu), mplkwargs, kwargs_defaults

class Figure(parameters.ParameterSet):
    def __init__(self, bundle, **kwargs):
        kwargs['context'] = 'plotting:figure'
        super(Figure, self).__init__(**kwargs)
        
        self.bundle = bundle
        #~ self.axes = []
        #~ self.axes_locations = []
        
    def _get_full_twig(self, twig):
        ti = self.bundle._get_by_search(twig, section='axes', class_name='Axes', return_trunk_item=True, ignore_errors=True)
        if ti is not None:
            return ti['twig']
        else:
            return None
    
    def add_axes(self, twig, loc=(1,1,1), sharex='_auto_', sharey='_auto_'):
        full_twig = self._get_full_twig(twig)
        
        #~ for key,append_value in zip(['axesrefs','axeslocs'],[full_twig,loc]):
            #~ arr = self.get_value(key)
            #~ arr.append(append_value)
            #~ self.set_value(arr)
        self['axesrefs'] += [full_twig]
        self['axeslocs'] += [loc]
        self['axessharex'] += [sharex if sharex=='_auto_' else self._get_full_twig(sharex)]
        self['axessharey'] += [sharey if sharey=='_auto_' else self._get_full_twig(sharey)]
        
    def remove_axes(self, twig):
        full_twig = self._get_full_twig(twig)
        #~ print "*** remove_axes", twig, full_twig
        
        if full_twig in self['axesrefs']:
            i = self['axesrefs'].index(full_twig)
            #~ print "*** remove_axes FOUND", i
            
            for k in ['axesrefs','axeslocs','axessharex','axessharey']:
                self[k] = [self[k][j] for j in range(len(self[k])) if j!=i]
            
        else:
            #~ print "*** remove_axes NOT FOUND"
            # reset any axessharex or axessharey that match
            for k in ['axessharex','axessharey']:
                new = ['_auto_' if c==full_twig else c for c in self[k]]
                self[k] = new
                
    def _get_for_axes(self, twig, key):
        full_twig = self._get_full_twig(twig)
        
        if full_twig in self['axesrefs']:
            ind = self['axesrefs'].index(full_twig)
            return self[key][ind]
        else: 
            return None

    def get_loc(self, twig):
        return self._get_for_axes(twig, 'axeslocs')
        
    def get_sharex(self, twig):
        return self._get_for_axes(twig, 'axessharex')
        
    def get_sharey(self, twig):
        return self._get_for_axes(twig, 'axessharey')
       
class Axes(parameters.ParameterSet):
    def __init__(self, bundle, **kwargs):
        kwargs['context'] = 'plotting:axes'
        super(Axes, self).__init__(**kwargs)
        
        self.bundle = bundle
        #~ self.plots = []

    def _get_full_twig(self, twig):
        ti = self.bundle._get_by_search(twig, context='plotting:plot', kind='ParameterSet', return_trunk_item=True, ignore_errors=True)
        if ti is not None:
            return ti['twig']
        else:
            return None
        
    def add_plot(self, twig):
        full_twig = self._get_full_twig(twig)
        
        self['plotrefs'] += [full_twig]
        
    def remove_plot(self, twig):
        full_twig = self._get_full_twig(twig)
        if full_twig in self['plotrefs']:
            i = self['plotrefs'].index(full_twig)
            
            for k in ['plotrefs']:
                self[k] = [self[k][j] for j in range(len(self[k])) if j!=i]  
