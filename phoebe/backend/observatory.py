"""
Convert a Body to an observable quantity.

.. autosummary::

    image
    contour
    ifm
    spectrum
    stokes
    astrometry

    
.. autosummary::
    
    compute
    observe


Extra functions to make plots or animations during computations:

.. autosummary::

    ef_binary_image
    ef_image
    
    
    
"""
# Modules from the standard library
import logging
import os
import itertools
import functools
import difflib
from collections import OrderedDict
# Third party dependencies: matplotlib and pyfits are try-excepted
import numpy as np
from numpy import pi, sqrt, sin, cos
from scipy.ndimage.interpolation import rotate as imrotate
from scipy.interpolate import griddata
from scipy import ndimage
try:
    import pylab as pl
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection,PolyCollection
    from matplotlib import animation
    import matplotlib as mpl
except ImportError:
    pass
try:
    import pyfits
except ImportError:
    pass
# Phoebe modules
from phoebe.utils import plotlib
from phoebe.utils import pergrams
from phoebe.utils import coordinates
from phoebe.utils import utils
from phoebe.utils import config
from phoebe.units import conversions
from phoebe.units import constants
from phoebe.parameters import parameters
from phoebe.parameters import datasets
from phoebe.algorithms import eclipse
from phoebe.algorithms import reflection
from phoebe.algorithms import marching
from phoebe.atmospheres import tools
from phoebe.atmospheres import passbands
from phoebe.atmospheres import limbdark
from phoebe.atmospheres import sed
from phoebe.atmospheres import spectra as modspectra
from phoebe.dynamics import keplerorbit
from phoebe.backend import decorators
from phoebe.backend import office
from phoebe.backend import plotting

# Ignore warnings raised by numpy, we'll be responsible for them ourselves        
np.seterr(all='ignore')

# Set up a logger
logger = logging.getLogger("OBSERVATORY")
logger.addHandler(logging.NullHandler())


#{ Computing observational quantities
    
def image(the_system, ref='__bol', context='lcdep',
            cmap=None, select='proj', background=None, vmin=None, vmax=None,
            beaming_alg='none', antialiasing=True,
            size=800, ax=None, savefig=False, nr=0, zorder=1, dpi=80,
            fourier=False, with_partial_as_half=True, coords='cartesian'):
    """
    Compute images of a system or make a 2D plot.
    
    You can make an image from basically any defined quantity in a mesh.
    
    To make a true image, you need intensities. By default, he *bolometric*
    intensities are used, but you can use the intensities from any observable,
    if you pass the correct ``ref`` and ``context``. Bolometric ones are the 
    default because this allows you to make an image of something for which no
    pbdeps are defined. 
    
    All the default parameters are set to make a true flux image of the system,
    in linear grayscale such that white=maximum flux and black=zero flux:
    
    >>> vega = phoebe.create.from_library('vega',create_body=True)
    >>> vega.set_time(0.)
    >>> fig_props, artist_props, patch = image(vega)
    
    .. image:: images/backend_observatory_image01.png 
       :scale: 20 %                                   
       :align: center                                 
    
    You can set C{select} to plot effective temperature (C{select='teff'},
    logg (C{select='logg'}) etc instead of projected flux. The colormap is
    adjustable via the C{cmap} keyword, and the background color via keyword
    ``background``. For some selections, there are
    smart default colormaps. E.g. for the effective temperature, the following
    two expressions yield the same result:
    
    >>> fig_props, artist_props, patch = image(vega, select='teff')
    >>> fig_props, artist_props, patch = image(vega, select='teff', cmap=pl.cm.hot,
    ...                                        background='0.7')
    
    .. image:: images/backend_observatory_image02.png 
       :scale: 20 %                                   
       :align: center                                 
    
    Setting C{select='rv'} will plot the radial velocity of the system, and then
    the colormap will automatically be changed to ``RdBu``, which means blue for
    velocities towards the observer, white for velocities in the plane of the
    sky, and red for velocities away from the observer. You can adjust the
    settings for the colorscale via ``vmin`` and ``vmax`` (if you set the
    limits, beware that the units of RV are Rsol/d!):
    
    >>> fig_props, artist_props, patch = image(vega, select='rv')
    >>> fig_props, artist_props, patch = image(vega, select='rv', vmin=-10, vmax=10)
    
    +---------------------------------------------------+---------------------------------------------------+
    | .. image:: images/backend_observatory_image03.png | .. image:: images/backend_observatory_image04.png |
    |   :width: 233px                                   |   :width: 233px                                   |
    |   :height: 233px                                  |   :height: 233px                                  |
    |   :align: center                                  |   :align: center                                  |
    +---------------------------------------------------+---------------------------------------------------+  
    
    More options for the ``select`` keyword are
    
    * ``proj``: projected flux (default)
    * ``teff``: effective temperature (K)
    * ``logg``: surface gravity (cm/s2 dex)
    * ``rv``: radial velocity (Rsol/d)
    * ``B``: strength of the magnetic field (G)
    * ``Bx``, ``By`` and ``Bz`` : magnetic field components (G)
    
    Setting C{select='teff'} gives you the option to color the map according
    to blackbody colors, via C{cmap='blackbody'}. Otherwise a heat map will
    be used. You cannot adjust the colorbar yourself when using blackbody
    colors, it will be scaled automatically between 2000K and 20000K. Hot
    objects will appear blue, cool objects will appear red, intermediate
    objects will appear white. Other scaling schemes with blackbodies are
    C{cmap='blackbody_proj'} and C{cmap='eye'}. The former uses the black body
    colors mapped from the temperature of each triangle, but will darken it
    according to the projected flux. The latter does something similar, but will
    saturate the colors to white at about half of the maximum intensity. This
    should make the object appear to glow (I find that it works better for
    hot objects).
    
    >>> fig, arts, patch = phoebe.image(vega, select='teff', cmap='blackbody')
    >>> fig, arts, patch = phoebe.image(vega, select='teff', cmap='blackbody_proj')
    >>> fig, arts, patch = phoebe.image(vega, select='teff', cmap='eye')
    
    +---------------------------------------------------+---------------------------------------------------+---------------------------------------------------+
    | .. image:: images/backend_observatory_image05.png | .. image:: images/backend_observatory_image06.png | .. image:: images/backend_observatory_image07.png |
    |   :width: 233px                                   |   :width: 233px                                   |   :width: 233px                                   |
    |   :align: center                                  |   :align: center                                  |   :align: center                                  |
    +---------------------------------------------------+---------------------------------------------------+---------------------------------------------------+    
    
    By default, a new figure is created, for which axes fill the whole canvas
    and the x and y-axis are completely removed. If you want to add an image
    to an existing axis, that is possible by giving that axis as an argument
    (:envvar:`ax`).
    In this case, the limits are not automatically set, so you need to set
    them manually. Luckily, this function returns a recommendation for the
    limits, as well as the collection of triangles themselves (see output below).
    The latter can be helpful if you want to add a colorbar. These options
    provide you with the utmost flexibility to incorporate the image of your
    Body in whatever customized plot or subplot you want. Beware that this
    function *does* set the scaling of the axis to be ``equal``, if you don't
    want that you'll need to readjust them yourself afterwards (see matplotlib
    documentation).
    
    >>> ax = pl.gca()
    >>> fig, arts, patch = phoebe.image(vega,ax=ax,background='white')
    >>> p = pl.xlim(fig['xlim'])
    >>> p = pl.ylim(fig['ylim'])
    >>> p = pl.xlabel("X-Distance [$R_\odot$]")
    >>> p = pl.ylabel("Y-Distance [$R_\odot$]")
    
    >>> cbar = pl.colorbar(patch)
    >>> cbar.set_label('Relative flux')
    
    +---------------------------------------------------+---------------------------------------------------+
    | .. image:: images/backend_observatory_image08.png | .. image:: images/backend_observatory_image09.png |
    |   :width: 233px                                   |   :width: 233px                                   |
    |   :align: center                                  |   :align: center                                  |
    +---------------------------------------------------+---------------------------------------------------+
    
    If you want to increase the number of pixels in an image, you need to set
    the keyword ``size``, which represent the number of pixels in the X or Y
    direction. Finally, for convenience, there is keyword ``savefig``. If you
    supply it will a string, it will save the figure to that file and close
    the image. This allows you to create and save an image of a Body with one
    single command.
    
    Finally, an experimental option is to compute the Fourier transform of
    an image instead of the normal image:
    
    >>> fig, arts, patch = image(vega,fourier=True)
    
    .. image:: images/backend_observatory_image10.png 
       :width: 233px
       :align: center                                 
    
    **Output**
    
    The output is a tuple with three entries:
    
        1. *figure properties* (dict), with entries:
            - *xlim*: (lower, upper) (solar radii)
            - *ylim*: (lower, upper) (solar radii)
        2. *artist properties* (dict), with entries:
            - *norm_proj*: normalisation constant to rescale mesh properties
              for use in the colorbar
            - *vmin*: minimum value of the rescaled color scale
            - *vmax*: maximum value of the rescaled color scale
        3. *patch collection* (matplotlib object)
    
    The latter is useful to add a colorbar to the plot, e.g. via::
    
        >>> figprop, artprop, pc = phoebe.image(vega,ax=ax,background='white')
        >>> plt.colorbar(pc)
    
    @param the_system: the Body to plot
    @type the_system: Body
    @param ref: reference of the intensities to use, if applicable
    @type ref: str or int
    @param context: context of the intensities
    @type context: str
    @param cmap: colormap to use
    @type cmap: matplotlib colormap or recognised str
    @param select: column name to use for plotting
    @type select: str
    @param background: axes background color
    @type background: matplotlib color
    @param vmin: minimum value for color scaling
    @type vmin: float
    @param vmax: maximum value for color scaling
    @type vmax: float
    @param size: size of the figure (in pixels if matplotlib's dpi=100)
    @type size: int
    @param ax: axes to plot in
    @type ax: matplotlib axes instance
    @return: figure propeties, artist properties, patch collection
    @rtype: dict, dict, matplotlib patch collection
    """
    # Default color maps and background depend on the type of dependables:
    if cmap is None and select == 'rv':
        cmap = pl.cm.RdBu_r
    elif cmap is None and select == 'teff':
        cmap = pl.cm.afmhot
        if background is None:
            background = '0.7'
    elif cmap is None and select == 'logg':
        cmap = pl.cm.gnuplot
        if background is None:
            background = '0.7'
    elif cmap is None and select[0] == 'B':
        cmap = pl.cm.jet
    elif cmap is None:
        cmap = pl.cm.gray
    # Default color for the background
    if background is None:
        background = 'k'
    # Default lower and upper limits for colors:
    vmin_ = vmin
    vmax_ = vmax
    
    # Get the parameterSet from which we need to take the intensities and other
    # information
    #if isinstance(ref, int):
    ref_ = ref
    ps, ref = the_system.get_parset(ref=ref_, context=context)
    #else:
    #    ps, ref = the_system.get_parset(ref=ref, context=context)
    if ps is None:
        possibilities = the_system.get_refs()
        close = difflib.get_close_matches(ref_, possibilities)
        raise ValueError(('obs with ref "{}" not found. Did you mean any of '
                          '"{}"? All possibilities: '
                          '{}').format(ref_,", ".join(close),", ".join(possibilities)))
    
    # We'll ask to compute the projected intensity,
    # because that is what we need for plotting. If it fails, various things
    # could have gone wrong, but the most likely one is that the user forgot to
    # set the time.
    logger.info('Making image of dependable set {}: plotting {}'.format(ref, select))    
    try:
        total_flux = the_system.projected_intensity(ref=ref,beaming_alg=beaming_alg,
                                      with_partial_as_half=with_partial_as_half)
    except ValueError as msg:
        raise ValueError(str(msg)+'\nPossible solution: did you set the time (set_time) of the system?')
    except AttributeError as msg:
        total_flux = 0.0
        logger.warning("Body has not attribute `projected_intensity', some stuff will not work")
    
    # Order the body's triangles from back to front so that they get plotted in
    # the right order.
    mesh = the_system.mesh
    sa = np.argsort(mesh['center'][:, 2])
    mesh = mesh[sa]
    
    # Now transform the coordinates to spherical coordinates
    # scales) if needed
    if coords == 'angular':
        globals = the_system.get_globals()
        if globals is None:
            raise ValueError('There are no global variables defined in this system: cannot compute angular coordinates.')
        distance = globals.request_value('distance', 'Rsol')
        origin = globals['ra'], globals['dec']
        mesh['center'] = np.array(keplerorbit.truecoords_to_spherical(mesh['center'],
                                         distance=distance, origin=origin, units='mas')).T
        mesh['triangle'][:,0:3] = np.array(keplerorbit.truecoords_to_spherical(mesh['triangle'][:,0:3],
                                         distance=distance, origin=origin, units='mas')).T
        mesh['triangle'][:,3:6] = np.array(keplerorbit.truecoords_to_spherical(mesh['triangle'][:,3:6],
                                         distance=distance, origin=origin, units='mas')).T
        mesh['triangle'][:,6:9] = np.array(keplerorbit.truecoords_to_spherical(mesh['triangle'][:,6:9],
                                         distance=distance, origin=origin, units='mas')).T
    
    x, y = mesh['center'][:, 0],mesh['center'][:, 1]
    
        
    
    # Initiate the figure: if a Fourier transform needs to be computed, we'll
    # create a small figure. Else we let the user decide. If the user supplied
    # axes when calling the function, we'll use that one (but set the axis'
    # background color and make sure the aspect is set to "equal").
    if fourier and ax is None:
        fig = pl.figure(figsize=(3, 3))
    elif ax is None:
        fig = pl.figure(figsize=(size/100., size/100.), dpi=dpi)
        
    if ax is None:
        # The axes should be as big as the figure, so that there are no margins
        ax = pl.axes([0, 0, 1, 1], axisbg=background, aspect='equal')
        fig = pl.gcf()
        # Make sure the background and outer edge of the image are black
        fig.set_facecolor(background)
        fig.set_edgecolor(background)
        axis_created = True
    else:
        ax.set_axis_bgcolor(background)
        ax.set_aspect('equal')
        axis_created = False
        
    # Set the values and colors of the triangles: there's a lot of possibilities
    # here: we can plot the projected intensity (i.e. as we would see it), but
    # we can also plot other quantities like the effective temperature, radial
    # velocity etc...
    cmap_ = None
    norm_proj = None
    if select == 'proj':
        colors = np.where(mesh['mu']>0, mesh['proj_'+ref] / mesh['mu'],0.0)       
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
    
    # Check for nans or all zeros, that usually means the user did something
    # wrong (OK, there's also a tiny chance that there's a bug somewhere)
    if np.all(values == 0):
        logger.warning("Image quantities are all zero, it's gonna be a dark picture...")
        if ref == '__bol':
            logger.warning("I see that the ref to be plotted is ref='__bol'. It is possible that no bolometric computations were done. Try setting ref=0 or the reference of your choice, or make sure bolometric computations are done. It is also possible that the object is totally eclipsed.")
    elif np.any(np.isnan(values)):
        logger.error("Discovered nans in values, it's gonna be an empty picture!")
        if ref == '__bol':
            logger.warning("I see that the ref to be plotted is ref='__bol'. It is possible that no bolometric computations were done. Try setting ref=0 or the reference of your choice, or make sure bolometric computations are done. It is also possible that the object is totally eclipsed.")
    
    # Collect the triangle objects for plotting
    patches = []
    if not cmap_ in ['blackbody_proj', 'eye'] and len(values.shape)==1:
        p = PolyCollection(mesh['triangle'].reshape((-1, 3, 3))[:, :, :2],
                             array=values,
                             closed=False,
                             antialiaseds=antialiasing,
                             edgecolors=cmap(colors),
                             facecolors=cmap(colors),
                             cmap=cmap, zorder=zorder)
    
    # When a RGB select color values is given
    elif not cmap_ in ['blackbody_proj', 'eye']:
        p = PolyCollection(mesh['triangle'].reshape((-1, 3, 3))[:, :, :2],
                             closed=False,
                             edgecolors=colors,
                             antialiaseds=antialiasing,
                             facecolors=colors, zorder=zorder)
    elif cmap_ == 'blackbody_proj':
        # In this particular case we also need to set the values for the
        # triangles first
        values = np.abs(mesh['proj_'+ref] / mesh['mu'])
        if 'refl_'+ref in mesh.dtype.names:
            values += mesh['refl_'+ref]
        scale = vmax if vmax is not None else 1.0
        values = (values / (scale*values.max())).reshape((-1, 1)) * np.ones((len(values), 4))
        values[:, -1] = 1.0
        colors = np.array([cmap(c) for c in colors]) * values 
        p = PolyCollection(mesh['triangle'].reshape((-1, 3, 3))[:, :, :2],
                             closed=False,
                             antialiaseds=antialiasing,
                             edgecolors=colors,
                             facecolors=colors,)
        
    elif cmap_ == 'eye':
        values = np.abs(mesh['proj_'+ref] / mesh['mu'])
        if 'refl_'+ref in mesh.dtype.names:
            values += mesh['refl_'+ref]
        keep = values > (0.5*values.max())
        values = values / values[keep].min()
        values = values.reshape((-1, 1)) * np.ones((len(values), 4))
        values[:, -1] = 1.0
        colors = np.array([cmap(c) for c in colors]) * values
        colors[colors > 1] = 1.0
        
        p = PolyCollection(mesh['triangle'].reshape((-1, 3, 3))[:, :, :2],
                             closed=False,
                             edgecolors=colors,
                             antialiaseds=antialiasing,
                             facecolors=colors)

    # Set the color scale limits
    if vmin is not None: vmin_ = vmin
    if vmax is not None: vmax_ = vmax
    p.set_clim(vmin=vmin_,vmax=vmax_)
    
    # Add the triangle plot objects to the axis, and set the axis limits to be
    # a tiny bit larger than the object we want to plot.
    ax.add_collection(p)
    
    # Derive the limits for the axis
    # (dont be smart when axis where given)
    offset_x = (x.min() + x.max()) / 2.0
    offset_y = (y.min() + y.max()) / 2.0
    margin = 0.01 * x.ptp()
    lim_max = max(x.max() - x.min(), y.max() - y.min())
    
    xlim = offset_x - margin - lim_max/2.0,offset_x + lim_max/2.0 + margin
    ylim = offset_y - margin - lim_max/2.0,offset_y + lim_max/2.0 + margin
    
    if axis_created:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        #pl.box(on=False)
    
    # Compute Fourier transform of image if needed
    if fourier:
        #-- save the above image, so that we can read it in to compute
        #   the 2D Fourier transform.
        pl.savefig('__temp.png',facecolor='k',edgecolor='k')
        pl.close()
        data = pl.imread('__temp.png')[:,:,0]
        data /= data.max()
        os.unlink('__temp.png')
        fsize = 4000
        FT = np.fft.fftn(data,s=[fsize,fsize])
        mag = np.abs(np.fft.fftshift(FT))
        freqs = np.fft.fftfreq(data.shape[0])
        freqs = np.fft.fftshift(freqs)
        #-- make image
        fig = pl.figure(figsize=(size/100.,size/100.))
        pl.axes([0,0,1,1],axisbg=background,aspect='equal')
        pl.imshow(np.log10(mag),vmin=0,vmax=np.log10(mag.max()),cmap=cmap)
        pl.xlim(fsize/2-50,fsize/2+50)
        pl.ylim(fsize/2-50,fsize/2+50)
        pl.xticks([]);pl.yticks([])
        fig.set_facecolor(background)
        fig.set_edgecolor(background)
    
        xlim, ylim = ax.get_xlim(),ax.get_ylim()
    
    if savefig == True:
        pl.savefig('image%06d.png'%(nr),facecolor='k',edgecolor='k')
        pl.close()
    elif savefig and os.path.splitext(savefig)[1]!='.fits':
        pl.savefig(savefig,facecolor='k',edgecolor='k')
        pl.close()
    elif savefig:
        pl.savefig('__temp.png',facecolor='k',edgecolor='k')
        pl.close()
        data = pl.imread('__temp.png')[:,:,0]
        os.unlink('__temp.png')
        d = the_system.get_distance()
        hdu = pyfits.PrimaryHDU(data)
        
        # for a simple linear projection, in RA and DEC (watch out: 'data axis 0' = y-axis!)
        hdu.header.update('CTYPE1',' ','')
        hdu.header.update('CTYPE2',' ','')
        
        # the central pixel of the image is used as the 'reference point'
        hdu.header.update('CRPIX1',data.shape[1]/2,'')
        hdu.header.update('CRPIX2',data.shape[0]/2,'')
        
        # no absolute location on the sky is needed for our purposes, so the 'world coordinate' of the 'reference point' is put at (0.,0.)
        hdu.header.update('CRVAL1',0.,'')
        hdu.header.update('CRVAL2',0.,'')
        
        # the angular size of 1 pixel = linear scale of one pixel / distance = linear scale of full image / number of pixels / distance
        resol1 = np.abs(xlim[0]-xlim[1])/d/data.shape[1]
        resol2 = np.abs(xlim[0]-xlim[1])/d/data.shape[0]
        hdu.header.update('CDELT1',resol1,'rad/pixel')
        hdu.header.update('CDELT2',resol2,'rad/pixel')
        
        # to be FITS-verifiable, angular coordinates should be in degrees (however ASPRO2 needs radians, so for now we use radians)
        hdu.header.update('CUNIT1',' ','should be deg, but is rad')
        hdu.header.update('CUNIT2',' ','should be deg, but is rad')
       
        hdulist = pyfits.HDUList([hdu])
        if os.path.isfile(savefig): os.unlink(savefig)
        hdulist.writeto(savefig)
        hdulist.close()
    
    figure_decorations = dict(xlim=xlim, ylim=ylim)
    artist_decorations = dict(norm_proj=norm_proj, vmin=vmin_, vmax=vmax_,
                              total_flux=total_flux)
    
    return figure_decorations, artist_decorations, p


def contour(system, select='B', res=300, prop=None, levels=None, ref=0, context='lcdep', **kwargs):
    """
    Draw contour lines on a Body.
    
    Possible contour lines:
        
        * ``longitude``: longitudinal lines
        * ``latitude``: latitudinal lines
        * ``B``: magnetic field lines
        * ``teff``: effective temperature lines
        * ``proj``: projected flux
        * ... (any column in the mesh)
    
    The dictionary ``prop`` is passed on to ``pl.clabel``, and can for example
    be ``prop = dict(inline=1, fontsize=14, fmt='%.0f G')``.
    
    **Example usage**
    
    First create a star like Vega.
    
    >>> vega = phoebe.create.from_library('vega',create_body=True)
    >>> vega.set_time(0.)

    The make three images, each time overlaying some contours. Note that are
    some difficulties with the longitudinal contour lines, because of the
    discontinuity at the pole and the longitudinal stitch. The quality of those
    plots will depend heavily on the resolution of the mesh (more than for other
    contour lines).
    
    >>> image(vega)
    >>> contour(vega, select='longitude')

    >>> image(vega)
    >>> contour(vega, select='latitude', colors='k', linewidths=2, linestyles='-')

    >>> image(vega)
    >>> contour(vega, select='teff', levels=[11000, 10500,10000,9500,9000],
    ...     cmap=plt.cm.hot, linewidths=2,
    ...     prop=dict(inline=1, fontsize=14, fmt='%.0f K'))

    +---------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+
    | .. image:: images/backend_observatory_contour01.png     | .. image:: images/backend_observatory_contour02.png  | .. image:: images/backend_observatory_contour03.png  |
    |   :width: 233px                                         |   :width: 233px                                      |   :width: 233px                                      |
    |   :height: 233px                                        |   :height: 233px                                     |   :height: 233px                                     |
    |   :align: center                                        |   :align: center                                     |   :align: center                                     |
    +---------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+    
    
    
    """
    # Set some defaults
    if prop is None:
        prop = dict()
    method = 'cubic'
    
    # Make a grid for the plane-of-sky coordinates: these are simply the x
    # and y coordinates of the mesh
    visible = system.mesh['visible']
    mesh = system.mesh[visible]
    x = mesh['center'][:,0]
    y = mesh['center'][:,1]
    xi = np.linspace(x.min(), x.max(), res)
    yi = np.linspace(y.min(), y.max(), res)
    
    # Get the values for the contours
    if select == 'B':
        z = sqrt(mesh['B_'][:,0]**2 + mesh['B_'][:,1]**2 + mesh['B_'][:,2]**2)
        if not 'colors' in kwargs:
            kwargs.setdefault('cmap', pl.cm.jet)
    elif select in ['latitude', 'longitude']:
        # For longitudinal and latitudinal meshes, we convert the original
        # Cartesian coordinates to spherical coordinates
        x_, y_, z_ = mesh['_o_center'].T
        rho, phi, theta = coordinates.cart2spher_coord(y_, x_, z_)
        phi = phi / np.pi * 180
        theta = theta / np.pi * 180 
        # Set some basic levels, to select which lines we want to plot
        if select == 'latitude':
            z = theta
            if levels is None:
                levels = np.arange(10,171,20.)
        else:
            method = 'linear'
            z = phi
            if levels is None:
                levels = np.arange(-160,161,20.)
    elif select == 'proj':   
        if not 'colors' in kwargs:
            kwargs.setdefault('cmap', pl.cm.gray)
        ps, ref = system.get_parset(ref=ref, context=context)
        #the_system.projected_intensity(ref=ref, with_partial_as_half=False)
        z = np.where(mesh['mu']>0, mesh['proj_'+ref] / mesh['mu'],0.0)       
        z = z/z.max()
    elif select == 'teff':
        if not 'colors' in kwargs:
            kwargs.setdefault('cmap', pl.cm.afmhot)
        z = mesh[select]
    elif select == 'rv':
        if not 'colors' in kwargs:
            kwargs.setdefault('cmap', pl.cm.RdBu_r)
        z = -mesh['velo___bol_'][:, 2] * 8.049861
    else:
        z = mesh[select]
    
    # Make a grid for the contour values
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method=method)

    # For longitudinal plots, try to remove the discontinuity. This is not ideal
    # but I don't know how else to do it.
    if select == 'longitude':
        phi_ = phi+0
        phi_[phi<0] = phi_[phi<0] + 360.
        zi_ = griddata((x, y), phi_, (xi[None,:], yi[:,None]), method=method)
        zi[(170<zi_) & (zi_<190)] = np.nan

    # plot the contours
    CS = pl.contour(xi, yi, zi, levels=levels, **kwargs)
    if prop:
        pl.clabel(CS, **prop)
    
    
def quiver(system, select='velo___bol_', **kwargs):
    """
    Draw vector quantities on a Body.
    
    Possible vector quantities lines:
        
        * ``B``: magnetic field lines
        * ``velo___bol_``: velocity field
        * ``normal_``: normal vectors
    
    """
    
    # Location of vectors
    visible = system.mesh['visible']
    mesh = system.mesh[visible]
    x = mesh['center'][:,0]
    y = mesh['center'][:,1]
    
    # Vector directions
    u = mesh[select][:,0]
    v = mesh[select][:,1]
    
    # Draw the arrows
    pl.quiver(x, y, u, v, **kwargs)
    
    

def surfmap(the_system,ref='__bol',context='lcdep',cut=0.96,
            cmap=None,select='proj',background=None,vmin=None,vmax=None,
            size=800,ax=None,savefig=False,nr=0,with_partial_as_half=True):
    """
    Compute images of a system or make a 2D plot.
    
    Very experimental, do not use this.
    
    All the default parameters are set to make a true flux image of the system,
    in linear grayscale such that white=maximum flux and black=zero flux.
    
    You can set C{select} to plot effective temperature (C{select='teff'}, logg
    (C{select='logg'}) etc instead of
    projected flux, and you can adjust the colormap via the C{cmap} keyword.
    
    Set the color of the background via C{background}.
    
    Setting C{select='rv'} will plot the radial velocity of the system, and then
    the colormap will automatically be changed to RdBu, which means blue for
    velocities towards the observer, white for velocities in the plane of the
    sky, and red for velocities away from the observer. **If you set the limits,
    beware that the units of RV are Rsol/d!**
    
    Setting C{select='teff'} gives you the option to color the map according to
    black body colors, via C{cmap='blackbody'}. Otherwise a heat map will
    be used. You cannot adjust the colorbar
    yourself when using blackbody colors, it will be scaled automatically between 2000K and 20000K. Hot
    objects will appear blue, cool objects will appear red, intermediate
    objects will appear white. Other scaling schemes with blackbodies are
    C{cmap='blackbody_proj'} and C{cmap='eye'}.
    
    If you want to add a colorbar yourself later, you can do so by giving the
    returned patch collection as an argument to matplotlib's C{colorbar} function.
    
    Size in pixels.
    
    >>> #image(time,the_system,savefig=True)
    
    @return: x limits, y limits, patch collection
    @rtype: tuple, tuple, patch collection
    """
    #-- default color maps and background depend on the type of dependables:
    if cmap is None and select=='rv':
        cmap = pl.cm.RdBu_r
    elif cmap is None and select=='teff':
        cmap = pl.cm.afmhot
        if background is None:
            background = '0.7'
    elif cmap is None and select[0]=='B':
        cmap = pl.cm.jet
    elif cmap is None:
        cmap = pl.cm.gray
    #-- default color for the background
    if background is None:
        background = 'k'
    #-- default lower and upper limits for colors:
    vmin_ = vmin
    vmax_ = vmax
        
    if isinstance(ref,int):
        ps,ref = the_system.get_parset(ref=ref,context=context)
    #-- to make an image, we need some info and we need to order it from
    #   back to front
    logger.info('Making image of dependable set {}: plotting {}'.format(ref,select))    
    try:
        the_system.projected_intensity(ref=ref,with_partial_as_half=with_partial_as_half)
    except ValueError as msg:
        raise ValueError(str(msg)+'\nPossible solution: did you set the time (set_time) of the system?')
    except AttributeError as msg:
        logger.warning("Body has not attribute `projected_intensity', some stuff will not work")
    mesh = the_system.mesh
    mesh = mesh[np.argsort(mesh['center'][:,2])]
    x,y = mesh['center'][:,0],mesh['center'][:,1]
    #-- initiate the figure
    if ax is None:
        fig = pl.figure(figsize=(size/100.,size/200.))
    if ax is None:
        #-- the axes should be as big as the figure, so that there are no margins
        ax = pl.axes([0,0,1,1],axisbg=background,aspect='equal')
        fig = pl.gcf()
        #-- make sure the background and outer edge of the image are black
        fig.set_facecolor(background)
        fig.set_edgecolor(background)
    else:
        ax.set_axis_bgcolor(background)
        ax.set_aspect('equal')
    #-- set the colors of the triangles
    cmap_ = None
    if select=='proj':
        colors = mesh['proj_'+ref]/mesh['mu']
        if 'refl_'+ref in mesh.dtype.names:
            colors += mesh['refl_'+ref]#/mesh['mu']
        colors /= colors.max()
        values = colors
        vmin_ = 0
        vmax_ = 1
    else:
        if select=='rv':
            values = -mesh['velo___bol_'][:,2]*8.049861
        elif select=='intensity':
            values = mesh['ld_'+ref+'_'][:,-1]
        elif select=='proj2':
            values = mesh['proj_'+ref]/mesh['mu']
            if 'refl_'+ref in mesh.dtype.names:
                values += mesh['refl_'+ref]#/mesh['mu']
        elif select=='Bx':
            values = mesh['B_'][:,0]
        elif select=='By':
            values = mesh['B_'][:,1]
        elif select=='Bz':
            values = mesh['B_'][:,2]    
        elif select=='B':
            values = np.sqrt(mesh['B_'][:,0]**2+mesh['B_'][:,1]**2+mesh['B_'][:,2]**2)
        else:
            values = mesh[select]
        if vmin is None: vmin_ = values[mesh['mu']>0].min()
        if vmax is None: vmax_ = values[mesh['mu']>0].max()
        colors = (values-vmin_)/(vmax_-vmin_)
        if cmap=='blackbody' or cmap=='blackbody_proj' or cmap=='eye':
            cmap_ = cmap
            cmap = plotlib.blackbody_cmap()
            vmin_ = 2000
            vmax_ = 20000
            colors = (values-vmin_)/(vmax_-vmin_)  
    #-- collect the triangle objects for plotting
    patches = []
    triangles = the_system.get_coords(type='spherical',loc='vertices')
    if not cmap_ in ['blackbody_proj','eye']:
        values_ = []
        colors_ = []
        sizes = np.ones(len(triangles))*np.inf
        coords = triangles.reshape((-1,3,3))[:,:,:2]
        side1 = coords[:,0]-coords[:,1]
        side2 = coords[:,1]-coords[:,0]
        side3 = coords[:,2]-coords[:,0]
        a = np.sqrt(np.sum(side1**2,axis=1))
        b = np.sqrt(np.sum(side2**2,axis=1))
        c = np.sqrt(np.sum(side3**2,axis=1))
        s = 0.5*(a+b+c)
        size = np.sqrt(abs( s*(s-a)*(s-b)*(s-c)))
        size = size*np.sin(coords[:,:,1].max()) #s[i] = size*np.sin(coords[:,1].max())
        
        decider = np.sort(size)[int(cut*len(size))]
        
        keep = size<decider
        for i,icoords in zip(np.arange(len(coords))[keep],coords[keep]):
            patches.append(Polygon(icoords,closed=True,edgecolor=cmap(colors[i])))
            values_.append(values[i])
            colors_.append(colors[i])
            
        p = PatchCollection(patches,cmap=cmap)
        #-- set the face colors of the triangle plot objects, and make sure
        #   the edges have the same color
        p.set_array(np.array(values_))
        p.set_edgecolor([cmap(c) for c in colors_])
        p.set_facecolor([cmap(c) for c in colors_])
    elif cmap_=='blackbody_proj':
        values = np.abs(mesh['proj_'+ref]/mesh['mu'])
        if 'refl_'+ref in mesh.dtype.names:
            values += mesh['refl_'+ref]
        values = (values/values.max()).reshape((-1,1))*np.ones((len(values),4))
        values[:,-1] = 1.
        colors = np.array([cmap(c) for c in colors])*values 
        
        for i,triangle in enumerate(triangles):
            patches.append(Polygon(triangle.reshape((3,3))[:,:2],closed=True,edgecolor=tuple(colors[i])))
        p = PatchCollection(patches,cmap=cmap)
        #-- set the face colors of the triangle plot objects, and make sure
        #   the edges have the same color
        p.set_edgecolor([tuple(c) for c in colors])
        p.set_facecolor([tuple(c) for c in colors])
    elif cmap_=='eye':
        values = np.abs(mesh['proj_'+ref]/mesh['mu'])
        if 'refl_'+ref in mesh.dtype.names:
            values += mesh['refl_'+ref]
        keep = values>(0.5*values.max())
        values = values/values[keep].min()
        values = values.reshape((-1,1))*np.ones((len(values),4))
        values[:,-1] = 1.
        colors = np.array([cmap(c) for c in colors])*values
        colors[colors>1] = 1.
        for i,triangle in enumerate(triangles):
            patches.append(Polygon(triangle.reshape((3,3))[:,:2],closed=True,edgecolor=tuple(colors[i])))
        p = PatchCollection(patches,cmap=cmap)
        #-- set the face colors of the triangle plot objects, and make sure
        #   the edges have the same color
        p.set_edgecolor([tuple(c) for c in colors])
        p.set_facecolor([tuple(c) for c in colors])
    #-- set the color scale limits
    if vmin is not None: vmin_ = vmin
    if vmax is not None: vmax_ = vmax
    p.set_clim(vmin=vmin_,vmax=vmax_)
    #-- add the triangle plot objects to the axis, and set the axis limits to
    #   be a tiny bit larger than the object we want to plot.
    ax.add_collection(p)
    #-- derive the limits for the axis
    # new style:
    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylim(0,np.pi)
        
    xlim,ylim = ax.get_xlim(),ax.get_ylim()
    if savefig==True:
        pl.savefig('image%06d.png'%(nr),facecolor='k',edgecolor='k')
        pl.close()
    elif savefig:
        pl.savefig(savefig,facecolor='k',edgecolor='k')
        pl.close()
    return xlim,ylim,p


# Helper function
def rotmatrix(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])


def prepare_bandwidth_smearing(ldgrid_file, passband, subdivide=10):
    """
    Prepare an atmosphere grid file for bandwidth smearking.
    
    This means:
        1. Subdivide the passband in a number of parts and temporarily add them
           to the library.
        2. Compute LD coeffs for all the parts (if they do not exist yet)
        3. Add those LD coeffs to the atmosphere file (if they do not exist yet) 
    """
    # Subdivide the passband
    responses, names = passband.subdivide_response(passband, parts=subdivide,\
                                                   add=True)
    
    # Add them to grid (won't do anything if they already exist)
    limbdark.compute_grid_ld_coeffs(ldgrid_file, passbands=names)
    
    

def bandwidth_smearing(fctn):
    """
    Apply bandwidth smearing to interferometry.
    
    This calls :py:func:`ifm` a couple of times, each time compute the
    visibilities in a different part of the passband. In the end, a weighted
    average of all visibilities is taken.
    """
    @functools.wraps(fctn)
    def do_bandwidth_smearing(system, *args, **kwargs):
        # In how many parts do we need to divide the passband?
        n_parts = kwargs.pop('bandwidth_smearing', 1)
        ref = kwargs['ref']
        
        # We need to look up the passband; for now hardcode it to 2MASS.KS
        parset, pref = system.get_parset(ref=ref, context='ifdep')
        passband = parset['passband']
        
        # For each port, compute the visibilities and such.
        output = []
        proj_int = np.ones(n_parts)
        flux = []
        transmission_weight = np.ones(n_parts)
        
        for n_part in range(n_parts):
            
            # Only use this subdivision when there's more than one part to
            # subdivide the bands over
            if n_parts>1:
                postfix = '{:04d}_{:04d}'.format(n_part, n_parts)
                this_ref = ref + postfix
                kwargs['ref'] = this_ref
                
                # Keep track of the total project intensity and relative
                # transmission of this part of the passband
                this_proj_int = system.projected_intensity(ref=kwargs['ref'])
                proj_int[n_part] = np.sum(np.array(this_proj_int))
                transmission_weight[n_part] = \
                         passbands.get_info([passband+postfix])['transmission']
        
                
            # Compute the ifm observables
            freq, base, pos_angle, visib, phase, \
                            ang_scale, ang_prof, this_flux = fctn(system, *args, **kwargs)
            flux.append(this_flux)
            output.append([visib, phase])
                        
        # Now compute all the averages...
        output = np.array(output)
        weights = proj_int*transmission_weight
        #print("in bandwidth smearing (obseratory.py) {} {}".format(proj_int, flux, this_flux))
        vis2, phase = np.average(output, axis=0, weights=weights)
        flux = np.array(flux).mean(axis=0)
        
        # That's it -- we just need to make sure to have the same return
        # signature as "ifm". Please change it in the future.
        return freq, base, pos_angle, vis2, phase, ang_scale, ang_prof, flux
    
    return do_bandwidth_smearing    
    
@bandwidth_smearing    
def ifm(the_system, posangle=0.0, baseline=0.0, eff_wave=None, ref=0,
        figname=None, bandwidth_smearing=1, keepfig=True):
    """
    Compute the Fourier transform of the system along a baseline.
    
    Algorithm steps:
    
        1. A flux-image is made using :py:func:`image` according to the
           prescriptions in the *ifdep* with reference :envvar:`ref`. This image
           is converted to a numpy array using :py:func:`fig2data <phoebe.utils.plotlib.fig2data>`.
        2. For every set of :envvar:`baseline`, :envvar:`posangle` and
           :envvar:`eff_wave` (the latter being computed from the passband if
           not available in the observations):
           
           1. **rotation**: the image array is rotated counter-clockwise
              according to the position angle (the same is done for the
              coordinates of the corners of the image, such that we know the
              absolute location of the object in space). The spatial coordinates
              are converted to angular coordinates by scaling with the distance. 
           2. **projection**: The image data is projected onto the X-axis in the
              newly rotated frame. We call this the ``angular_profile``.
           3. **amplitude**: The ``total_flux`` in the image is computed by
              summing up the flux in every pixel. In principle, this should be
              rotation independent, but because of pixel-interpolation during
              the rotation, there is probably some variation on this. The total
              flux factor is import for normalisation of the visibilities, 
              specifically when combining visibilities of different components.
              The output is normalised, but the unnormalised visibilities can
              be computed easily by scaling with the total flux.
           4. **Fourier transform**: The Discrete Fourier transform is computed
              for the observed spatial frequencies. These are computed from the
              baselines.
    
    If you give a single positional angle as a float instead of an array,
    the whole profile will be computed. Otherwise, only selected baselines
    will be computed, to e.g. match observed data.
    
    .. note::
    
        Thanks to M. Hillen. He is not responsible for bugs or errors.
    """
    # Information on what to compute
    data_pars, ref = the_system.get_parset(ref)
    passband = data_pars['passband']
    
    # Make sure we can cycle over the baselines and posangles
    single_baseline = False
    if not hasattr(baseline,'__iter__'):
        single_baseline = True
        baseline = [baseline]
    if not hasattr(posangle,'__iter__'):
        posangle = [posangle]
    
    # Prepare output
    frequency_out = []
    baseline_out = []
    posangle_out = []
    visibility_out = []
    phase_out = []
    angular_scale_out = []
    angular_profile_out = []
    total_flux_out = []
    
    # Set effective wavelength to the one from the passband if not given
    # otherwise
    if eff_wave is None:
        eff_wave = passbands.get_info([passband])['eff_wave'][0]*np.ones(len(posangle))
    logger.info("ifm: cyclic frequency to m at lambda~%.4g angstrom"%(eff_wave.mean()))
    
    # Make an image if necessary, but in any case retrieve it's dimensions
    if figname is None:
        figname = 'ifmfig_temp.png'
        #keep_figname = False
        #xlims,ylims,p = image(the_system,ref=ref,savefig=figname)
        #data = pl.imread(figname)[:,:,0]
        #os.unlink(figname)
        figdec, artdec, p = image(the_system,ref=ref, dpi=100, antialiasing=False, context='ifdep')
        xlims = figdec['xlim']
        ylims = figdec['ylim']
        data = np.array(plotlib.fig2data(pl.gcf(), grayscale=True),float)
        total_flux = artdec['total_flux']
        pl.close()
        
    else:
        keep_figname = True
        figname,xlims,ylims = figname
        data = pl.imread(figname)[:,:,0]
        total_flux = 1.0
    coords =np.array([[i,j] for i,j in itertools.product(xlims,ylims)])
    
    #-- cycle over all baselines and position angles
    #d = the_system.as_point_source()['coordinates'][2]#list(the_system.params.values())[0].request_value('distance','Rsol')
    d = the_system.get_distance()
    #dpc = conversions.convert('Rsol','pc',d)
    dpc = d*2.253987922034374e-08
    
    # Note: rotation might go faster but perhaps less precise if replaced with:
    #Nx, Ny = image.shape
    #margin = int(Nx / 2.0)
    # X, Y = np.ogrid[-Nx/2:Nx/2, -Ny/2:Ny/2]
    #rotated_image = np.zeros((Nx + margin, Ny + margin))
    #rotated_image[margin/2:-margin/2, margin/2:-margin/2] = data
    
    prev_pa = None
    for nr, (bl, pa, wl) in enumerate(zip(baseline, posangle, eff_wave)):        
        
        if keepfig:
            xlims,ylims,p = image(the_system,ref=ref,context='ifdep',savefig='{}_{:05d}.fits'.format(keepfig,nr))
            figdec,artdec,p = image(the_system,ref=ref,context='ifdep')
            xlims = figdec['xlim']
            ylims = figdec['ylim']
            pl.gca().set_autoscale_on(False)
            
            #-- add the baseline on the figure
            x_toplot = np.linspace(xlims[0]-0.5*abs(xlims[1]-xlims[0]),xlims[1]+0.5*abs(xlims[1]-xlims[0]),100)
            y_toplot = np.zeros_like(x_toplot)
            x_toplot_ = x_toplot*np.cos(pa/180.*pi)
            y_toplot_ = x_toplot*np.sin(pa/180.*pi)
            pl.plot(x_toplot_,y_toplot_,'r-',lw=2)
            vc = bl*np.sin(pa/180.*np.pi)
            uc = bl*np.cos(pa/180.*np.pi)
            npix = data.shape[0]
            resol = np.abs(xlims[0]-xlims[1])/d
            #resol = conversions.convert('rad','mas',resol)/npix
            resol =  resol / (2 * np.pi) * 360. * 3600. / npix
            an_text = (r'PA={:.2f}$^\circ$' +'\n'
                       r'$\lambda$={:.0f}$\AA$' +'\n'
                       r'B={:.0f}m' +'\n'
                       r'U,V=({:.1f},{:.1f}) m' + '\n'
                       r'{:d} pix' +'\n'+ '{:.3g} mas/pix')
            pl.annotate(an_text.format(pa,eff_wave[nr],bl,uc,vc,npix,resol),
                        (0.95,0.95),va='top',ha='right',
                        xycoords='axes fraction',color='r',size=20)
        
        # Rotate counter clockwise by angle in degrees, and recalculate the
        # values of the corners
        # We add a shortcut here not to repeat the rotation if we're at the
        # same position angle as before. That's nice because then we can easily
        # compute the whole profile as a function of baseline.
        if pa!=prev_pa:
            data_ = imrotate(data,-pa,reshape=True,cval=0., order=0) # was -pa
        
        prev_pa = pa
        # Rotation might go faster if replaced with -- but we need to take a bigger
        # array!
        #X_, Y_ = np.cos(-theta)*X - np.sin(-theta)*Y,\
        #         np.sin(-theta)*X + np.cos(-theta)*Y
        #X_ = np.array(X_,int) + Nx/2 + margin/2 - 1
        #Y_ = np.array(Y_,int) + Ny/2 + margin/2 - 1
        #data__ = new_image[X_, Y_]
        
        
        if keepfig and keepfig is not True: # then it's a string
            pl.figure()
            pl.subplot(111,aspect='equal')
            pl.imshow(data_,origin='image')
            pl.savefig('{}_{:05d}_rot.png'.format(keepfig,nr))
            pl.close()
        coords2 = np.dot(coords,rotmatrix(-pa/180.*np.pi)) # was -pa
        xlims = coords2[:,0].min(),coords2[:,0].max()
        ylims = coords2[:,1].min(),coords2[:,1].max()
        #-- project onto X-axis
        signal = data_.sum(axis=0)
        #-- compute Discrete Fourier transform: amplitude and phases
        x = np.array(np.linspace(xlims[0],xlims[1],len(signal)),float)
        x = x/d # radians
        #x = conversions.convert('rad','as',x) # arseconds
        x = x / (2 * np.pi) * 360. * 3600.
        #x -= x[0]
        if single_baseline and bl==0: 
            nyquist = 0.5/(x[1]-x[0])
            f0 = 0.01/x.ptp()
            fn = nyquist/25.
            df = 0.01/x.ptp()
            logger.info('ifm: single baseline equal to 0m: computation of entire profile')
        else:
            #f0 = conversions.convert('m','cy/arcsec',bl,wave=(wl,'angstrom'))*2*np.pi
            f0 = conversions.baseline_to_spatialfrequency(bl, wl)
            fn = f0
            df = 0
            #logger.info('ifm: computation of frequency and phase at f0={:.3g} cy/as (lam={:.3g}AA)'.format(f0,wl))
            if keepfig and keepfig is not True:
                pl.annotate('f={:.3g} cy/as\n d={:.3g} pc'.format(f0,dpc),(0.95,0.05),va='bottom',ha='right',xycoords='axes fraction',color='r',size=20)
                pl.figure()
                pl.plot(x*1000.,signal,'k-')
                pl.grid()
                pl.xlabel('Coordinate [mas]')
                pl.ylabel("Flux")
                pl.savefig('{}_{:05d}_prof.png'.format(keepfig,nr))
                #np.savetxt('prof.dat', np.column_stack([x*1000, signal]))
                pl.close()
        #-- to take band pass smearing into account, we need to let the
        #   wavelength vary over the passband, and add up all the
        #   Fourier transforms but weighted with the SED intensity
        #   times the transmission filter.
        signal = signal / signal.sum() * total_flux
        #signal = signal/signal.sum()
        #f1_,s1_ = pergrams.DFTpower(x,signal,full_output=True,
        #                    f0=f0,fn=fn,df=df)
        s1 = pergrams.DFT(x, signal, f0)
        f1 = f0
        
        # We could add a check for the Nyquist frequency here, but that should
        # be always fine for realistic observations
        #print('frequency = {:.3g}, nyquist = {:.3g} ({:.3f}%)'.format(f0, 0.5/(x[1]-x[0]),f0/0.5*(x[1]-x[0])))
        
        if np.angle(s1)<-np.pi:
            print("angular diameter = {} mas, angle={}".format(x[signal>0].ptp()*1000, np.angle(s1), np.angle(s1)- x.ptp()*pi*f1))
        
        s1_vis = np.abs(s1)
        s1_phs = np.angle(s1)
        #-- correct cumulative phase
        #s1_phs = s1_phs - x.ptp()*pi*f1
        #s1_phs = (s1_phs % (2*pi))# -pi
        #b1 = conversions.convert('cy/arcsec','m',f1,wave=(wl,'angstrom')) / (2*np.pi)
        b1 = conversions.spatialfrequency_to_baseline(f1, wl)
        if keepfig and keepfig is not True:
            pl.savefig('{}_{:05d}.png'.format(keepfig,nr))
            pl.close()
                
        #-- append to output
        frequency_out.append(f1)
        baseline_out.append(b1)
        posangle_out.append(pa)
        visibility_out.append(s1_vis**2)
        phase_out.append(s1_phs)
        angular_scale_out.append(x)
        angular_profile_out.append(signal)
        total_flux_out.append(total_flux)
        
    if single_baseline and baseline[0]==0:
        frequency_out = frequency_out[0]
        baseline_out = baseline_out[0]
        posangle_out = posangle_out[0]
        visibility_out = visibility_out[0]
        phase_out = phase_out[0]
        angular_scale_out = angular_scale_out[0]
        angular_profile_out = angular_profile_out[0]
        total_flux_out = total_flux_out[0]
    else:
        frequency_out = np.array(frequency_out)
        baseline_out = np.array(baseline_out)
        posangle_out = np.array(posangle_out)
        visibility_out = np.array(visibility_out)
        phase_out = np.array(phase_out)
        total_flux_out = np.array(total_flux_out)
        #angular_scale_out = np.array(angular_scale_out)
        #angular_profile_out = np.array(angular_profile_out)
    
    return frequency_out,baseline_out,posangle_out,\
           visibility_out,phase_out,\
           angular_scale_out,angular_profile_out, total_flux_out


def spectrum(the_system, obs, pbdep, rv_grav=True):
    """
    Compute the spectrum of a system.
    
    Method: weighted sum of synthesized spectra in a limited wavelength range
    corresponding to the local effective temperature and gravity:
    
    .. math::
    
        S(\lambda) = \sum_{i=1}^N A_i F_i S_i(\lambda,T_\mathrm{eff},\log g, v_\mathrm{rad}[, z])
    
    The spectra themselves are not normalised: in fact, they are small pieces
    of high-resolution spectral energy distributions. The local temperature
    and gravity are reflected both in the depth of the lines, as well as in
    the continuum. Limb-darkening is taken into account by taking limb
    darkening coefficients in a photometric passband in the vicinity of the
    wavelength range (the user is responsible for this). Limb darkening
    coefficients are thus not wavelength dependent within one spectrum or
    spectral line profile, in the current implementation. Also, the spectral
    line profile is independent of limb angle.
    
    Instrumental broadening is assumed to be Gaussian, and can only be applied
    when the resolution of the generated large is finer than the instrumental
    broadening.
    
    Microturbulence can be taken into account by setting :envvar:`vmicro` in
    the corresponding datasets. This effectively adds another Gaussian
    convolution to the spectrum. In principle, this is also not OK, since the
    microturbulence changes the equivalent width of the line. This is not the
    case here. Better is to take microturbulence into account during grid
    computations.
    
    Returns: wavelength ranges, total (unnormalized) spectrum, total continuum.
    
    You can get the normalised spectrum by dividing the flux with the
    continuum.
    
    We probably need to following info:
    
        * time stamp (and possibly exposure time)
        * dispersion type (lin, log, variable) --> taken care of by wavelength
          definition in obs
        * wavelength span (wmin,wmax for lin and log, the whole array for
          variable dispersion) --> taken care of by wavelength definition in obs
        * sampling power --> taken care of by R in spdep
        * resolving power or dispersion, depending on dispersion type
          --> R in spobs
        * passband (if any) -- passband present
        * auxiliary information that is not used but may be set for reference
          (grating, angle rotation, ..., whatever) --> not necessary
    
    .. note::
        
        We do not take into account the influence of limb-darkening in
        the line shape. Although it seems to work out pretty well in
        practice, this is wrong.
              
    
    """
    ref = obs['ref']
    mesh = the_system.mesh
    
    # Wavelength info
    if not 'wavelength' in obs or not len(obs['wavelength']):
        loaded = obs.load(force=False)
    try:
        wavelengths = obs.request_value('wavelength', 'AA').ravel()
    except:
        raise ValueError("Either wavelength is not in dataset, or it's not an array. Perhaps this is a weird location to throw this error, and this should be checked upon array creation")
    
    # Instrumental and stellar field info
    Robs = obs.get('R', None)
    
    # Intrinsic width of the profile
    Rmod = pbdep.get('R_input', 0.0)
    vmicro = pbdep.get('vmicro', 2.0)
    vmacro = pbdep.get('vmacro', 0.0)
    depth = pbdep.get('depth', 0.4)
    alphaT = pbdep.get('alphaT', 0.0)
    
    # System velocity offset
    vgamma = obs.get('vgamma_offset', 0.0)
    
    # Information on dependable set: we need the limb darkening function and
    # the method
    ld_model = pbdep['ld_func']
    method = pbdep['method']
    profile = pbdep['profile']
    
    keep = the_system.mesh['mu'] >= 0
    
    # Set the central wavelength "wc".
    wc = (wavelengths[0]+wavelengths[-1])/2.
    
    # Broaden the range of the wavelengths a bit, so that we are sure that also
    # neighbouring lines are taken into account. We clip the spectrum
    # aftwards. This "a bit" is taken to be the wavelength range corresponding
    # to the maximum RV range in the model
    rv_range = the_system.mesh['velo___bol_'][:,2].ptp()
    wave_extension = conversions.convert('Rsol/d', 'nm', rv_range, wave=(wc, 'nm')) - wc
    
    # Define wavelength template for model. It depends on the given wavelength
    # range, the extension and resolution
    wavelengths_orig = wavelengths
    w0 = wavelengths.min() - wave_extension
    wn = wavelengths.max() + wave_extension
    if Rmod == 0:
        deltaw = np.median(np.diff(wavelengths))
    else:
        deltaw = wc / Rmod
    wavelengths = np.arange(w0, wn + deltaw, deltaw)
    
    # Check if there is any flux. If we're not seeing the star, we can easily
    # compute the spectrum: it's zero! Hihihi (hysterical laughter)!
    the_system.projected_intensity(ref=ref, method='numerical')
    keep = the_system.mesh['proj_'+ref] > 0
    
    if not np.sum(keep):
        logger.info('no spectrum {} synthesized, zero flux received'.format(ref))
        return wavelengths, np.zeros(len(wavelengths)), np.ones(len(wavelengths))
    
    cc_ = constants.cc / 1000.
    
    teff_local = the_system.mesh['teff'][keep]
    
    # Get mean teff, this is important for the change in depth of Gaussian
    # lines
    try:
        main_parset = the_system.params.values()[0]
        teff_mean = main_parset['teff']
    except IndexError:
        logger.critical("Cannot figure out reference temperature, set to T=10000K")
        teff_mean = 10000.
        alpha_T = 0.0
    except KeyError:
        logger.critical("Cannot figure out reference temperature, set to T=10000K")
        teff_mean = 10000.
        alpha_T = 0.0
    
    # Numerical method with profiles from a precomputed grid:
    if method == 'numerical' and not profile == 'gauss':    
        
        # Get limb angles
        mus = the_system.mesh['mu']
        keep = (mus > 0) & (the_system.mesh['partial'] | the_system.mesh['visible'])
        mus = mus[keep]
        
        # Negating the next array gives the partially visible things, that is
        # the only reason for defining it.
        visible = the_system.mesh['visible'][keep]
        
        # Compute normalised intensity using the already calculated limb
        # darkening coefficents. They are normalised so that center of disk
        # equals 1. Then these values are used to compute the weighted sum of
        # the spectra.
        logger.info(("using limbdarkening law {} - spectra interpolated "
                     "from grid {}").format(ld_model, profile))
        
        ld_func = getattr(limbdark, 'ld_{}'.format(ld_model))
        Imu = ld_func(mus, the_system.mesh['ld_' + ref][keep].T)
        teff, logg = the_system.mesh['teff'][keep], the_system.mesh['logg'][keep]
        
        # Interpolate (fitters can go outside of grid)
        if profile == 'blackbody':
            spectra = np.ones((2,len(wavelengths),len(teff)))
            for i,iteff in enumerate(teff):
                spectra[1,:,i] = sed.blackbody(wavelengths/10., iteff)
        else:
            try:
                spectra = modspectra.interp_spectable(profile, teff, logg, wavelengths)
                                
            except IndexError:
                logger.error(("no spectrum synthesized (outside of grid "
                          "({}<=teff<={}, {}<=logg<={}), zero flux "
                          "received").format(teff.min(), teff.max(),
                                             logg.min(), logg.max()))
                          
                return wavelengths, np.zeros(len(wavelengths)), np.ones(len(wavelengths))
        
        
        # Compute the spectrum
        proj_intens = spectra[1] * mus * Imu * the_system.mesh['size'][keep]
        rad_velos = -the_system.mesh['velo___bol_'][keep, 2]
        rad_velos = rad_velos * 8.04986111111111 # from Rsol/d to km/s
        rad_velos += vgamma
        logger.info('Systemic radial velocity = {:.3f} km/s'.format(vgamma))
        logger.info('synthesizing spectrum using %d faces (RV range = %.6g to %.6g km/s)'%(len(proj_intens),rad_velos.min(),rad_velos.max()))

        total_continum = 0.
        total_spectrum = 0.
        
        # gravitational redshift:
        if rv_grav:
            rv_grav = tools.gravitational_redshift(the_system)
            rad_velos += rv_grav[keep]
            
        for i,rv in enumerate(rad_velos):
            
            # Not inline
            total_spectrum += tools.doppler_shift(wavelengths, rv,
                                                  flux=spectra[0,:,i]*proj_intens[:,i])
            total_continum += tools.doppler_shift(wavelengths, rv,
                                                  flux=proj_intens[:,i])
                                                          
    # Numerical computation with Gaussian profile
    elif method == 'numerical':
        
        # Derive intrinsic width of the profile
        sigma = conversions.convert('km/s', 'AA', vmicro, wave=(wc, 'AA')) - wc
        logger.info('Intrinsic width of the profile: {} AA ({} km/s)'.format(sigma, vmicro))
        template = 1.00 - depth*np.exp( -(wavelengths-wc)**2/(2*sigma**2))
        proj_intens = the_system.mesh['proj_'+ref][keep]
        rad_velos = -the_system.mesh['velo___bol_'][keep,2]
        rad_velos = rad_velos * 8.04986111111111 # from Rsol/d to km/s
        
        rad_velos += vgamma
        logger.info('Systemic radial velocity = {:.3f} km/s'.format(vgamma))
        sizes = the_system.mesh['size'][keep]
        logger.info('synthesizing Gaussian profile using %d faces (RV range = %.6g to %.6g km/s)'%(len(proj_intens),rad_velos.min(),rad_velos.max()))
        total_continum = np.zeros_like(wavelengths)
        total_spectrum = 0.
        
        
        
        #-- gravitational redshift:
        #if rv_grav:
        #    rv_grav = 0#generic.gravitational_redshift(the_system)
        template_ = template
        for i,(pri,rv,sz) in enumerate(zip(proj_intens,rad_velos,sizes)):
            
            #spec = pri*sz*tools.doppler_shift(wavelengths,rv+rv_grav,flux=template)
            if alphaT > 0:
                template_ = 1.00 - (1.0+alphaT) * (teff_local[i]/teff_mean)*(1-template)
            
            wave_out1 = wavelengths * (1 + rv/cc_)
            spec = pri*sz*np.interp(wavelengths,wave_out1,template_)
            
            total_spectrum += spec
            total_continum += pri*sz
    
    # Analytical computation
    elif method == 'analytical' and profile == 'gauss':
        #-- For the analytical computation, we require a linear limb darkening
        #   law
        if not ld_model =='linear':
            raise ValueError("Analytical computation of spectrum requires a 'linear' limb-darkening model (not '{}')".format(idep['ld_func']))
        epsilon = pbdep['ld_coeffs'][0]
        vrot = the_system.mesh['velo___bol_'][:,2].max() # this is vsini!
        vrot = vrot * 8.04986111111111 # from Rsol/d to km/s
        logger.info('analytical rotational broadening with veq=%.6f km/s'%(vrot))
        #teff = the_system.params['star']['teff']
        #logg = the_system.params['star'].request_value('logg','[cm/s2]')
        #spectra = limbdark.interp_spectable('atlas',[teff],[logg],wavelengths)
        #spectrum = spectra[0]/spectra[1]
        sigma = conversions.convert('km/s','AA', vmicro, wave=(wc,'AA'))-wc
        logger.info('Intrinsic width of the profile: {} AA'.format(sigma))
        template = 1.00 - depth*np.exp( -(wavelengths-wc)**2/(2*sigma**2))
        wavelengths, total_spectrum = tools.rotational_broadening(wavelengths,
                              template, vrot, stepr=-1, epsilon=epsilon)
        total_continum = np.ones_like(wavelengths)

    #-- convolve with instrumental profile if desired
    if Robs is not None and Robs > 0:
        instr_fwhm = wc/Robs
        
        logger.info('Convolving spectrum with instrumental profile of FWHM={:.3f}AA (R={})'.format(instr_fwhm, Robs))
        try:
            total_spectrum = tools.broadening_instrumental(wavelengths,
                       total_spectrum/total_continum, width=instr_fwhm,
                       width_type='fwhm')
            total_spectrum *= total_continum
        except ValueError:
            logger.info("Cannot convolve spectrum, resolution too low wrt instrumental broadening")
    
    # Microturbulent velocity is approximated as an extra Gaussian broadening
    # in the case of precomputed grid spectra. Obviously, this is not correct:
    # the precomputed grid should implement vmicro since also the line depths
    # are influenced. In other words, this is a quick-n-dirty approximation.
    if vmicro > 0 and profile != 'gauss':
        dlam_micro = vmicro/constants.cc*1000*wc
        logger.info('Convolving spectrum with microturbulent profile dlam_mic={:.3f}AA'.format(dlam_micro))
        try:
            total_spectrum = tools.broadening_instrumental(wavelengths,
                       total_spectrum/total_continum, width=dlam_micro,
                       width_type='sigma')
            total_spectrum *= total_continum
        except ValueError:
            logger.error("Cannot convolve spectrum, resolution too low wrt microturbulent broadening")
    
    # Interpolate onto original wavelength grid
    total_spectrum = np.interp(wavelengths_orig, wavelengths, total_spectrum)
    total_continum = np.interp(wavelengths_orig, wavelengths, total_continum)
    
    # Scale the projected intensity with the distance
    globals_parset = the_system.get_globals()
    if globals_parset is not None:
        distance = globals_parset.request_value('distance', 'Rsol')
        total_spectrum /= distance**2
        total_continum /= distance**2
    
    return wavelengths_orig, total_spectrum, total_continum


def stokes(the_system, obs, pbdep, rv_grav=True):
    r"""
    Compute the Stokes profiles of a system.
    
    What you need to do is to calculate the Zeeman effect, i.e. the shift
    between the right circularly polarised line and the left circularly
    polarised line. Then you use your I profile, you shift it by the Zeeman
    shift and you substract the 2 shifted (right and left) profiles. This
    gives you the local Stokes V profile.
    
    For the calculation of the Zeeman effect you will need to know the
    value of gamma = Acos(abs(bz_xi)/b) and of b (as well as theta, phi and
    the projected area) for each point on your star. That will allow you to
    calculate the local Stokes V profile, and then you integrate over the
    visible stellar hemisphere.
    
    This is what I do, assuming Gaussian profiles:
    
    First, I compute the Zeeman splitting for each surface element:
    
    .. math::
    
        \Delta\nu_\mathrm{Zeeman,i} = \frac{-g_lq_eB_i}{4\pi m_e}\quad \mathrm{[Hz]}
        
    with :math:`g_l` the Lande factor, and :math:`q_e` and :math:`m_e` the fundamental electron charge and mass.
    Next, I compute a simple Gaussian line profile for each surface element
    :math:`i` (:math:`G^0_i(\lambda,0,\sigma)`), which contributes to the
    Stokes I profile, and the two shifted profiles :math:`G^+_i(\lambda,+\Delta\lambda_\mathrm{zeeman},\sigma)`,
    and :math:`G^-_i(\lambda,-\Delta\lambda_\mathrm{zeeman},\sigma)` for the
    computation of the Stokes V profile.
    
    The Stokes I profile is simply the sum of all the :math:`G^0_i` contributions:
    
    .. math::
    
        I = \sum_i A_iI_{\mu,\lambda} G^0_i
    
    Here :math:`A_i` is the area of the ith surface element, and
    :math:`I_{\mu,\lambda}` is the projected intensity of the ith surface element.
    The Stokes V profile is computed as
    
    .. math::
    
        V = \sum_i \frac{1}{2} A_iI_{\mu,\lambda} \cos\theta_i (G^-_i-G^+_i)
    
    or, when ``weak_field=True`` we'll use the weak-field approximation
    
    .. math::
    
        V(\nu) = \sum_i -\cos\theta_i \Delta\nu_{Z,i}\frac{d G^0_i}{d\nu}
    
    with
    
    .. math::
        
        \cos\theta = \frac{\vec{B}\cdot\vec{s}}{Bs}
        
    and :math:`s` the line-of-sight vector (which is in the minus Z-direction).
    Thus the latter is equivalent to
    
    .. math::
        
        \cos\theta = \frac{\vec{B}\cdot\vec{B_z}}{B B_z}
    
    Similarly, the Stokes Q and U profiles are computed as
    
    .. math::
    
        Q(\nu) = -\sum_i\frac{1}{4}\sin^2\theta_i\cos^22\chi_i \Delta\nu^2_{Z,i}\frac{d^2 G^0_i}{d^2\nu}\\
        U(\nu) = -\sum_i\frac{1}{4}\sin^2\theta_i\sin^22\chi_i \Delta\nu^2_{Z,i}\frac{d^2 G^0_i}{d^2\nu}
        
    
    The method of computation is exactly the same when using spectral profiles
    from a grid, except the wavelength-dependent intensity is now a function
    of many more parameters:
    
    .. math::
        I_{\mu,\lambda} \longrightarrow I_{\mu,\lambda,T_\mathrm{eff},\log g,z,v_\mathrm{rad},\ldots}
    
    .. note::
        
        Thanks to C. Neiner and E. Alecian. They are not responsible for any bugs or errors.
    
    """
    ref = obs['ref']
    mesh = the_system.mesh
    
    # Wavelength info
    if not 'wavelength' in obs or not len(obs['wavelength']):
        loaded = obs.load(force=False)
    wavelengths = obs.request_value('wavelength', 'AA').ravel()
    
    # Instrumental info
    R = obs.get('R', None)
    vmacro = obs.get('vmacro', 0.0)
    vrad_obs = obs.get('vgamma', 0.0)
    
    # Intrinsic width of the profile
    vmicro = pbdep.get('vmicro', 5.0)
    depth = pbdep.get('depth', 0.4)
    alphaT = pbdep.get('alphaT', 0.0)
    
    # Profiles to compute, and method to use
    do_V = do_Q = do_U = False
    if 'V' in obs['columns']:
        do_V = True
    if 'Q' in obs['columns']:
        do_Q = True
    if 'U' in obs['columns']:
        do_U = True
    logger.info("Computing V ({}), Q ({}), U ({})".format(do_V, do_Q, do_U))
    
    # Information on dependable set: we need the limb darkening function, the
    # method, the glande factor and the weak field approximation boolean.
    ld_model = pbdep['ld_func']
    method = pbdep['method']
    glande = pbdep['glande']
    profile = pbdep['profile']
    weak_field = pbdep['weak_field']
    keep = the_system.mesh['mu'] <= 0
    
    # Set the central wavelength "wc".
    wc = (wavelengths[0]+wavelengths[-1])/2.
    
    # If we're not seeing the star, we can easily compute the spectrum: it's
    # zero! Muhuhahaha (evil laughter)!
    if not np.sum(keep):
        logger.info('Still need to compute (projected) intensity')
        the_system.intensity(ref=ref)
    
    # Check if there is any flux
    the_system.projected_intensity(ref=ref, method='numerical')
    keep = the_system.mesh['proj_'+ref] > 0
    
    if not np.sum(keep):
        logger.info('no spectropolarimetry synthesized zero flux received')
        return wavelengths, np.zeros(len(wavelengths)),\
               np.zeros(len(wavelengths)), np.zeros(len(wavelengths)),\
               np.zeros(len(wavelengths)), np.ones(len(wavelengths))
    
    
    # Magnitude of magnetic field and angle towards the LOS (if there is any)
    B = coordinates.norm(mesh['B_'][keep], axis=1) * 1e-4 # and convert to Tesla 
    
    if np.any(B != 0.0):
        cos_theta = coordinates.cos_angle(mesh['B_'][keep],
                                          np.array([[0.0, 0.0,-1.0]]), axis=1)
    else:
        cos_theta = np.zeros(len(mesh))
    
    # Create some shortcut arrays to avoid extensive repetition of calculations
    sin2theta = 1 - cos_theta**2
    cos_chi = coordinates.cos_angle(mesh['B_'][keep],
                                    np.array([[1.0, 0.0, 0.0]]), axis=1)
    cos2chi = cos_chi**2
    sin2chi = 1. - cos2chi
    cos22chi = (cos2chi - sin2chi)**2
    sin22chi = 1. - cos22chi
    
    # Zeeman splitting in angstrom  (qe is negative but somewhere my B field has
    # wrong sign?)
    delta_nu_zeemans = -glande * constants.qe * B/ (4*np.pi*constants.me)
    delta_nu_zeemans2 = delta_nu_zeemans**2
    delta_v_zeemans = (wc * 1e-10) * delta_nu_zeemans / 1000. # from cy/s to km/s
    
    # Radial velocities
    rad_velos = -the_system.mesh['velo___bol_'][keep, 2]
    rad_velos = rad_velos * 8.04986111111111 # from Rsol/d to km/s
    nus = constants.cc/wavelengths*1e10 # from AA to Hz
    
    # Correct radial velocities for vgamma in observation set
    rad_velos = rad_velos + vrad_obs 
    
    # Report which approximation we use
    approx_msg = weak_field and 'Weak-field' or 'No weak-field'
    logger.info('{} approximation'.format(approx_msg))
    
    if method == 'numerical' and not profile == 'gauss':
        # Get limb angles
        mus = the_system.mesh['mu']
        keep = (mus > 0) & (the_system.mesh['partial'] | the_system.mesh['visible'])
        mus = mus[keep]
        
        # Negating the next array gives the partially visible things, that is
        # the only reason for defining it.
        visible = the_system.mesh['visible'][keep]
        
        # Compute normalised intensity using the already calculated limb
        # darkening coefficents. They are normalised so that center of disk
        # equals 1. Then these values are used to compute the weighted sum of
        # the spectra.
        logger.info('using limbdarkening law {} - spectropolarimetry interpolated from grid {}'.format(ld_model,pbdep['profile']))
        Imu = getattr(limbdark,'ld_{}'.format(ld_model))(mus, the_system.mesh['ld_'+ref][keep].T)
        teff, logg = the_system.mesh['teff'][keep], the_system.mesh['logg'][keep]
            
        # Interpolate (fitters can go outside of grid)    
        spectra = modspectra.interp_spectable(profile, teff, logg, wavelengths)
            
        proj_intens = spectra[1] * mus * Imu * the_system.mesh['size'][keep]
        logger.info('synthesizing spectropolarimetry using %d faces (RV range = %.6g to %.6g km/s)'%(len(proj_intens),rad_velos.min(),rad_velos.max()))

        total_continum = 0.0
        stokes_I = 0.0
        stokes_V = 0.0
        stokes_Q = 0.0
        stokes_U = 0.0
        
        # Gravitational redshift:
        if rv_grav:
            rv_grav = 0.0 # generic.gravitational_redshift(the_system)
        for i,rv in enumerate(rad_velos):
            rvz = delta_v_zeemans[i]
            total_continum += tools.doppler_shift(wavelengths, rv+rv_grav,\
                                                  flux=proj_intens[:, i])
            
            # Compute left and right shifted profile, as well as the usual
            # intensity profile
            specm = tools.doppler_shift(wavelengths, rv+rv_grav-rvz,
                                      flux=spectra[0, :, i] * proj_intens[:, i])
            specp = tools.doppler_shift(wavelengths, rv+rv_grav+rvz,
                                      flux=spectra[0, :, i] * proj_intens[:, i])  
            spec = tools.doppler_shift(wavelengths, rv+rv_grav,
                                      flux=spectra[0, :, i] * proj_intens[:, i])
            stokes_I += spec
            
            # Stokes V (in weak field approximation or not)
            if do_V and weak_field:
                stokes_V -= cos_theta[i] * delta_nu_zeemans[i] * utils.deriv(nus, spec)
            elif do_V:
                stokes_V += cos_theta[i] * (specm-specp) / 2.0

    # This case: numerical but Gaussian profile
    elif method == 'numerical':
        # Derive intrinsic width of the profile
        sigma = conversions.convert('km/s', 'AA', vmicro, wave=(wc, 'AA')) - wc
        logger.info('Intrinsic width of the profile: {} AA ({} km/s)'.format(sigma, vmicro))
        
        template = 1.00 - depth * np.exp( -(wavelengths-wc)**2/(2*sigma**2))
        proj_intens = the_system.mesh['proj_'+ref][keep]
        sizes = the_system.mesh['size'][keep]
        
        logger.info('synthesizing Gaussian profile using %d faces (sig= %.2e AA,RV range = %.6g to %.6g km/s)'%(len(proj_intens),sigma,rad_velos.min(),rad_velos.max()))
        total_continum = np.zeros_like(wavelengths)
        stokes_I = 0.0
        stokes_V = 0.0
        stokes_Q = 0.0
        stokes_U = 0.0
        
        # Gravitational redshift:
        if rv_grav:
            rv_grav = 0.0 # generic.gravitational_redshift(the_system)
        
        rad_velosw = conversions.convert('km/s', 'AA', rad_velos, wave=(wc, 'AA')) - wc
        
        cc_ = constants.cc / 1000.
        teff_local = the_system.mesh['teff'][keep]
        
        try:
            main_parset = the_system.params.values()[0]
            teff_mean = main_parset['teff']
        except IndexError:
            logger.critical("Cannot figure out reference temperature, set to T=10000K")
            teff_mean = 10000.
            alpha_T = 0.0
        except KeyError:
            logger.critical("Cannot figure out reference temperature, set to T=10000K")
            teff_mean = 10000.
            alpha_T = 0.0
        
        iterator = zip(proj_intens,rad_velos,sizes,B,cos_theta)
        for i, (pri, rv, sz, iB, costh) in enumerate(iterator):
            
            rvz = delta_v_zeemans[i]
            #-- first version
            #spec  = pri*sz*tools.doppler_shift(wavelengths,rv+rv_grav,flux=template)
            #specm = pri*sz*tools.doppler_shift(wavelengths,rv+rv_grav-rvz,flux=template)
            #specp = pri*sz*tools.doppler_shift(wavelengths,rv+rv_grav+rvz,flux=template)
            
            # Correct the template for the temperature:
            template_ = 1.00 - (1.0+alphaT) * (teff_local[i]/teff_mean)*(1-template)
            
            # First version but inline
            wave_out1 = wavelengths * (1+(rv+rv_grav)/cc_)
            wave_out2 = wavelengths * (1+(rv+rv_grav-rvz)/cc_)
            wave_out3 = wavelengths * (1+(rv+rv_grav+rvz)/cc_)
            spec = pri*sz*np.interp(wavelengths,wave_out1,template_)
            specm = pri*sz*np.interp(wavelengths,wave_out2,template_)
            specp = pri*sz*np.interp(wavelengths,wave_out3,template_)
            
            # We can compute Stokes V in weak field approximation or not
            if do_V and weak_field:
                stokes_V -= costh * delta_nu_zeemans[i] * utils.deriv(nus, spec)
            elif do_V:
                stokes_V += costh * (specm - specp) / 2.0
            
            #-- second version: weak field approximation
            #mytemplate = pri*sz*(1.00 - depth*np.exp( -(wavelengths-wc-rad_velosw[i])**2/(2*sigma**2)))
            #stokes_V -= costh*delta_nu_zeemans[i]*utils.deriv(nus,mytemplate)
            #- third version: weak field approximation
            #stokes_V -= costh*delta_nu_zeemans[i]*utils.deriv(nus,spec)
            
            #-- Stokes Q and U: this must be in weak field approximation for now
            if do_Q or do_U:
                sec_deriv = utils.deriv(nus, utils.deriv(nus, spec))
            if do_Q:
                stokes_Q -= 0.25*sin2th*cos22chi*delta_nu_zeemans2[i]*sec_deriv
            if do_U:
                stokes_U -= 0.25*sin2th*cos22chi*delta_nu_zeemans2[i]*sec_deriv
            
            stokes_I += spec
            
            total_continum += pri*sz      
        
        #logger.info("Zeeman splitting: between {} and {} AA".format(min(conversions.convert('Hz','AA',delta_nu_zeemans,wave=(wc,'AA'))),max(conversions.convert('Hz','AA',delta_nu_zeemans,wave=(wc,'AA')))))
        #logger.info("Zeeman splitting: between {} and {} Hz".format(min(delta_nu_zeemans/1e6),max(delta_nu_zeemans/1e6)))
        logger.info("Zeeman splitting: between {} and {} km/s".format(min(delta_v_zeemans), max(delta_v_zeemans)))
    else:
        raise NotImplementedError
    
    # Convolve with instrumental profile if desired
    if R is not None:
        instr_fwhm = wc/R
        logger.info('Convolving spectrum with instrumental profile of FWHM={:.3f}AA'.format(instr_fwhm))
        try:
            stokes_I = tools.broadening_instrumental(wavelengths,
                       stokes_I/total_continum, width=instr_fwhm,
                       width_type='fwhm')        
            stokes_I *= total_continum
        
            stokes_V = tools.broadening_instrumental(wavelengths,
                       1-stokes_V/total_continum, width=instr_fwhm,
                       width_type='fwhm')        
            stokes_V = (1-stokes_V)*total_continum
        except:
            logger.error("Cannot convolve spectrum, resolution too low wrt instrumental broadening")
            
    return wavelengths, stokes_I, stokes_V, stokes_Q, stokes_U, total_continum
    

def radial_velocity(the_system, obs, rv_grav=True):
    """
    Compute a Body's radial velocity
    """
    ref = obs['ref']
    mesh = the_system.mesh    
    keep = the_system.mesh['mu'] >= 0
    
    # If we're not seeing the star, we can easily compute the spectrum: it's
    # zero! Hihihi (hysterical laughter)!
    if not np.sum(keep):
        logger.info('Still need to compute (projected) intensity')
        the_system.intensity(ref=ref)
    
    # Check if there is any flux    
    the_system.projected_intensity(ref=ref, method='numerical')
    keep = the_system.mesh['proj_'+ref] > 0
    
    if not np.sum(keep):
        logger.info('no radial velocity {} computed, zero flux received'.format(ref))
        return np.nan
    
    # Get limb angles
    mus = the_system.mesh['mu']
    keep = (mus > 0) & (the_system.mesh['partial'] | the_system.mesh['visible'])
    mus = mus[keep]
    
    # Negating the next array gives the partially visible things, that is
    # the only reason for defining it.
    visible = the_system.mesh['visible'][keep]
    proj_intens = the_system.mesh['proj_'+ref][keep]
    rad_velos = -the_system.mesh['velo___bol_'][keep,2]
    rad_velos = rad_velos * 8.04986111111111 # from Rsol/d to km/s
    if rv_grav:
        rv_grav = tools.gravitational_redshift(the_system)
        #print rad_velos
        rad_velos = rad_velos + rv_grav[keep]
        #print rad_velos
        #import matplotlib.pyplot as plt
        #plt.plot(rad_velos)
        #plt.plot(rv_grav[keep])
        #plt.plot(rad_velos + rv_grav[keep])
        #plt.show()
    
    return np.average(rad_velos, weights=proj_intens)

def astrometry(system, obs, pbdep, index):
    """
    Compute a body's apparent coordinates on the sky.
    
    For more information, see :py:func:`apparent_coordinates <phoebe.dynamics.keplerorbit.apparent_coordinates>`.
    """
    myglobals = system.get_globals()
    ra = myglobals['ra']
    dec = myglobals['dec']
    pmra = myglobals['pmra']
    pmdec = myglobals['pmdec']
    distance = myglobals['distance']
    distance_Rsol = distance*constants.pc/constants.Rsol
    epoch = myglobals['epoch']
    
    # Get photocenter of the system, but remove the distance (this is taken
    # care of inside apparent_coordinates
    target_position = system.as_point_source(ref=pbdep['ref'])['photocenter'].reshape((3,1))
    target_position[2] -= distance_Rsol
    observer_position = obs['time'][index], obs['eclx'][index],\
                        obs['ecly'][index], obs['eclz'][index]
    t0 = None#obs['t0']
    
    output = keplerorbit.apparent_coordinates(distance, ra, dec, pmra, pmdec,
                                observer_position,target_position,
                                epoch=epoch, t0=t0)
    return output




#}
#{ Input/output

def add_bitmap(system,image_file,select='teff',minval=None,maxval=None,res=1,
               update_intensities=False, shift=0, white_as_transparent=True):
    """
    Add a pattern from a bitmap figure to a mesh column.
    
    At this point this function is for playing. Perhaps some day it can be
    useful.
    
    Use PNG! Not JPG!
    
    Shift horizontally in degrees
    
    Ideally, the image is twice as wide as it is high.
    """
    #-- get the coordinates in the original frame of reference.
    r,phi,theta = system.get_coords()
    #-- read in the data
    data = pl.imread(image_file)[::res,::res]
    #   convert color images to gray scale
    if len(data.shape)>2:
        data = data.mean(axis=2).T
    else:
        data = data.T
    data = data[::-1]
    
    # Shift horizontal
    if shift:
        #data = np.roll(data, int(0.25*data.shape[0]), axis=0)
        phi = phi - shift / 180. * np.pi
        phi[phi>2*np.pi] = phi[phi>2*np.pi]-2*np.pi
        phi[phi<0] = phi[phi<0]+2*np.pi
    #-- normalise the coordinates so that we can map the coordinates
    PHI = (phi+np.pi)/(2*np.pi)*data.shape[0]
    THETA = (theta)/np.pi*data.shape[1]
    vals = np.array(ndimage.map_coordinates(data,[PHI,THETA],mode='nearest'),float)
    #-- fix edges of image
    #vals[PHI>0.99999*PHI.max()] = 1.0
    #vals[PHI<0.00001*PHI.max()] = 1.0
    
    #-- rescale values between 0 and 1 so that we can map them between
    #   minval and maxval
    if minval is None: minval = system.mesh[select].min()
    if maxval is None: maxval = system.mesh[select].max()
    
    #-- don't map white values, but let the starlight shine through!
    if white_as_transparent:
        keep = vals<0.99
    else:
        keep = np.ones(len(vals), bool)
    vals = (vals-vals.min())
    vals = vals/vals.max()
    vals = vals*(maxval-minval) + minval
    #-- and replace the values in the particular column
    system.mesh[select][keep] = vals[keep]
    if update_intensities:
        system.intensity()
    logger.info("Added bitmap {}".format(image_file))
           


        
def extract_times_and_refs(system, params, tol=1e-8):
    """
    Automatically extract times, references and types from a BodyBag.
    
    This function files the keys C{time}, C{types} and C{refs} in the
    parameterSet with context C{params}.
    
    If times are already set with a list, this function does nothing. This
    function automatically fills the value if C{time='auto'}, with the
    information inside the enabled datasets. If C{time='all'}, the enabled
    state will be disregarded and all datasets are taken into account.
    
    @param system: Body to derive time points of
    @type system: Body
    @param params: ParameterSet to fill in
    @type params: ParameterSet
    @param tol: tolerance limit for accepting two different time points as equal
    @type tol: float
    """
    # Do we really need to set anything?
    dates = params['time']
    if not (isinstance(dates, str)):
        return None
    times = [] # time points
    types = [] # types
    refs  = [] # references
    samps = [] # sampling rates
    
    # Some datasets don't really need any numerical meshing. We'll keep those
    # separated because they require a different structure to hold: we need
    # type, reference and timepoints
    no_mesh_required = dict()
    
    found_obs = False
    
    # Collect the times of the data, the types and refs of the data: walk
    # through all the parameterSets, if they represent observations, load the
    # dataset and collect the times. The reference and type for each
    # parameterset is the same. Correct the time points to compute something on
    # for exposure time and sampling rates: we add time points, which in the end
    # should be then averaged over.
    for parset in system.walk():
        
        # Skip stuff that is not a DataSet
        if not isinstance(parset, datasets.DataSet):
            continue
        
        # Skip DataSets that are not representing observations (e.g. the
        # synthetics)
        if not parset.context[-3:] == 'obs':
            continue
        found_obs = True
        
        # Figure out if we need to compute a mesh to deal with this obs
        require_mesh = True
        category = parset.get_context()[:-3]
        dep_parset = system.get_parset(context=category+'dep', ref=parset['ref'])[0]
        if category == 'rv' and 'method' in dep_parset and dep_parset['method'] == 'dynamical':
            require_mesh = False
        elif category == 'am' and 'method' in dep_parset and dep_parset['method'] == 'dynamical':
            require_mesh = False
        
        # If the dataset is not enabled, forget about it (if dates == 'auto')
        if dates == 'auto' and not parset.get_enabled():
            continue
        loaded = parset.load(force=False)

        # Retrieve exposure times
        if 'exptime' in parset and hasattr(parset, '__len__') and len(parset['exptime']):
            exps = parset['exptime']
        else:
            exps = [0] * len(parset)
        
        # Sampling rates
        if 'samprate' in parset and len(parset['samprate']):
            samp = parset['samprate']
        else:
            samp = [1] * len(parset)

        # Now the user could have given phases instead of times. The definition
        # of "phase" is quite system-morphology-dependent.
        if 'phase' in parset['columns'] and not 'time' in parset['columns']:    
            period, t0, shift = system.get_period()
            
            if np.isinf(period):
                raise ValueError("Don't know how to unphase observations")
            
            mytimes = (parset['phase'] * period) + t0
        
        else:
            mytimes = parset['time']
        
        # If a mesh is required add the info to the right lists
        if require_mesh:
            for itime, iexp, isamp in zip(mytimes, exps, samp):
                iexp_d = iexp/(24*3600.)
                times.append(np.linspace(itime - iexp_d/2., itime + iexp_d/2.0, isamp+1)[:-1]+(iexp_d/2./isamp))

                refs.append([parset['ref']] * isamp)
                types.append([parset.context] * isamp)
                samps.append([isamp]*isamp)
        
        # If a mesh is not required, add the info to the no_mesh_required dict
        else:
            if not category in no_mesh_required:
                no_mesh_required[category] = dict()
            no_mesh_time = []
            no_mesh_samp = []
            for itime, iexp, isamp in zip(mytimes, exps, samp):
                iexp_d = iexp/(24*3600.)
                no_mesh_time.append(np.linspace(itime - iexp_d/2., itime + iexp_d/2.0, isamp+1)[:-1]+(iexp_d/2./isamp))
                no_mesh_samp.append([isamp]*isamp)
            no_mesh_time = np.array(no_mesh_time).ravel()
            no_mesh_samp = np.array(no_mesh_samp).ravel()
            no_mesh_required[category][parset['ref']] = dict(time=no_mesh_time,
                                                           samprate=no_mesh_samp)
        
        # Put the parameterSet in the state we found it
        if loaded: parset.unload()
    
    # Next we allow for time points to be considered "the same", so that we
    # don't need to recompute stuff for very small differences in time (at least
    # small agains the expected variations). We cannot test "true" equality
    # because of possible numerical rounding (e.g. when reading or writing
    # files). We test which times are "nearly" equal.
    
    # But first we put all the times together, and sort them chronologically
    try:
        times = np.hstack(times)
    except ValueError as msg:
        raise ValueError(("Failed to derive at which points the system needs "
                         "to be computed. Perhaps the obs are not DataSets? "
                         "(original message: {})").format(str(msg)))
    except IndexError as msg:
        # If we happen to only have stuff that doesn't require a mesh,
        # we can get away with it
        if no_mesh_required:
            params['time'] = []
            params['refs']= []
            params['types'] = []
            params['samprate'] = []
            return no_mesh_required
        if not found_obs:
            raise ValueError(("Failed to derive at which points the system needs "
                          "to be computed. I can't find any obs attached to "
                          "the system, check if they are added properly. "
                          "(original message: {})").format(str(msg)))
        else:
            raise ValueError(("Failed to derive at which points the system needs "
                          "to be computed. I found obs attached to the system, "
                          "but perhaps they contain no time information (or an "
                          "empty time column)? "
                          "(original message: {})").format(str(msg)))
    
    sa    = np.argsort(times)
    types = np.hstack(types)[sa]
    refs  = np.hstack(refs)[sa]
    samps = np.hstack(samps)[sa]
    times = times[sa]
    
    # For each time point, keep a list of stuff that needs to be computed
    labl_per_time = [] # observation ref (uuid..)
    type_per_time = [] # observation type (lcdep...)
    time_per_time = [] # times of observations
    samp_per_time = [] # sampling rates
    
    for i, t in enumerate(times):
        
        # If the time is the first or different from the previous one: remember
        # the time and start a list of refs and types
        if i == 0 or np.abs(t - time_per_time[-1]) >= tol:
            time_per_time.append(times[i])
            labl_per_time.append([refs[i]])
            type_per_time.append([types[i]])
            samp_per_time.append([samps[i]])
        
        # Else, append the refs and times to the last time point
        elif labl_per_time[-1][-1] != refs[i]:
            labl_per_time[-1].append(refs[i])
            type_per_time[-1].append(types[i])
            samp_per_time[-1].append(samps[i])
        
        else:
            # Don't know what to do here: also append or continue?
            #continue
            labl_per_time[-1].append(refs[i])
            type_per_time[-1].append(types[i])
            samp_per_time[-1].append(samps[i])
    # And fill the parameterSet!
    params['time'] = time_per_time
    params['refs']= labl_per_time
    params['types'] = type_per_time
    params['samprate'] = samp_per_time

    return no_mesh_required


    



def compute_one_time_step(system, i, time, ref, type, samprate, reflect, nreflect,
                          circular, heating, beaming, params, ltt, save_result,
                          extra_func, extra_func_kwargs):
    """
    Compute a system on one given time step.
    
    Do not *ever* use this, unless you know what you're doing. The parameters
    in this function are derived in :py:func:compute.
    
    You are welcome to browse through the source code though. Have a glass of
    wine, your favorite easy chair. And of course, a compact disc playing on
    your home stereo. So flip of your shoes, lift your feet up. Lean back, and
    enjoy the melodies.
    
    Because you shouldn't use this function, I will not describe the parameters
    in details. Muhaha.
    """
    # Unsubdivide to prepare for this step (if necessary)
    subdiv_num = params['subdiv_num']
    subdiv_alg = params['subdiv_alg']
    eclipse_alg = params['eclipse_alg']
    irradiation_alg = params['irradiation_alg']
    
    if subdiv_num:
        system.unsubdivide()
    
    # Execute some pre-processing steps if necessary
    system.preprocess(time)
    
    # Clear previous reflection effects if necessary (not if reflect==1 because
    # one might want to keep the reflection results in circular orbits or if they
    # otherwise do not change!)
    
    if reflect is True:
        system.clear_reflection()
    
    # Set the time of the system: this will put everything in the right place,
    # and compute the necessary physical quantities.
    system.set_time(time, ref=ref, beaming_alg=beaming)
    
    # Compute intensities: it is possible that this is already taken care of
    # in set_time. It doesn't hurt to do it again, but this might be optimized.
    if circular and (not beaming == 'full') and not ref in system.__had_refs:
        system.intensity(ref=ref, beaming_alg=beaming)
        system.__had_refs.append(ref)
    elif (not circular) or (beaming == 'full'):
        system.intensity(ref=ref, beaming_alg=beaming)
    
    # Compute pblum
    if i == 0:
        system.compute_pblum_or_l3()
    
    # Compute reflection effect (maybe just once, maybe always). If this is done
    # we need to update the intensities
    update_intensity = False
    if (reflect is True or heating is True) or (i == 0 and (reflect == 1 or heating == 1)):
        reflection.mutual_heating(*system.get_bodies(), heating=heating,
                                  reflection=reflect, niter=nreflect,
                                  irradiation_alg=irradiation_alg)
        update_intensity = True
    
    # Recompute the intensities, temperatures might have changed due to
    # reflection)
    if update_intensity:
        system.intensity(ref=ref, beaming_alg=beaming)
    
    # Detect eclipses/horizon, and remember the algorithm that was chosen. It
    # will be re-used after subdivision
    ecl, found_partial = choose_eclipse_algorithm(system, algorithm=eclipse_alg)

    # If necessary, subdivide and redetect eclipses/horizon
    for k in range(subdiv_num):
        if found_partial is False:
            continue
        logger.debug('Subdividing stage {}/{}'.format(k+1, subdiv_num))
        system.subdivide(threshold=0, algorithm=subdiv_alg)
        ecl_, found_partial = choose_eclipse_algorithm(system, algorithm=ecl)
    
    # Compute the necessary observables at this time step
    had_refs = [] # we need this for the ifm, so that we don't compute stuff too much
    for itype, iref, isamp in zip(type, ref, samprate):
        if itype[:-3] == 'if':
            itype = 'ifmobs' # would be obsolete if we just don't call it "if"!!!
            if iref in had_refs:
                continue
            had_refs.append(iref)
        logger.info('Calling {} for ref {}'.format(itype[:-3], iref))
        getattr(system, itype[:-3])(ref=iref, time=time,
                                    correct_oversampling=isamp,
                                    beaming_alg=beaming,
                                    save_result=save_result)

    # Call extra funcs if necessary
    for ef, kw in zip(extra_func, extra_func_kwargs):
        ef(system, time, i, **kw)
        
    # Execute some post-processing steps if necessary
    system.postprocess(time)
    
    
def animate_one_time_step(i, system, times, refs, types, samprate, reflect, nreflect,
            circular, heating, beaming, params, ltt, save_result, extra_func,
            extra_func_kwargs, anim):
    """
    Compute one time step and animate it.
    """
    compute_one_time_step(system, i, times[i], refs[i], types[i], samprate[i], reflect, nreflect,
                          circular, heating, beaming, params, ltt, save_result, extra_func,
                          extra_func_kwargs)
    anim.draw()
    # Close the window once the animation has finished.
    if i==(len(times)-1) and not anim.repeat and anim.close_after_finish:
        logger.info("Animation finished; closing window")
        pl.close()
    
   
    
    


@decorators.mpirun
def compute(system, params=None, extra_func=None, extra_func_kwargs=None,
            animate=None, **kwargs):
    """
    Automatically compute dependables of a system to match the observations.
    
    This is typically want you want to do if you have some data and want to
    compute a model generating those data. The independent variables (times,
    wavelengths, UV-coords etc...), references and types at which to compute
    anything, will be derived from the observations attached to the
    :envvar:`system`.
    
    Detailed configuration of the computations is provided via the optional
    :ref:`compute <parlabel-phoebe-compute>` parameterSet, set via the keyword
    :envvar:`params`::
    
            time auto   --   phoebe Compute observables of system at these times
            refs auto   --   phoebe Compute observables of system at these times
           types auto   --   phoebe Compute observables of system at these times
         heating False  --   phoebe Allow irradiators to heat other Bodies
            refl False  --   phoebe Allow irradiated Bodies to reflect light
        refl_num 1      --   phoebe Number of reflections
             ltt False  --   phoebe Correct for light time travel effects
      subdiv_alg edge   --   phoebe Subdivision algorithm
      subdiv_num 3      --   phoebe Number of subdivisions
     eclipse_alg auto   --   phoebe Type of eclipse algorithm
     beaming_alg full   --   phoebe Type of beaming algorithm

    But for convenience, all parameters in this parameterSet can also be
    given as keyword arguments (kwargs).
        
    You can give an optional :ref:`mpi <parlabel-phoebe-mpi>` parameterSet for
    multithreading across different cores and/or machines.
    
    The keywords ``extra_func`` and ``extra_func_kwargs`` accept a list of extra
    functions to run after each time step. You can use this for example to plot
    an image of a binary system as you compute through the orbit. See example
    functions such as :py:func:`ef_binary_image` for example signatures of these
    functions. It is perfectly possible that user-defined functions (i.e. those
    not residing in this module) will not be callable when using MPI. If you're
    looking for functionality that automatically processes some information
    before and after computations (e.g. setting correlated parameters), you need
    to use the pre- and postprocessing tools from the Body.
    
    Live animation is supported via :envvar:`animate` (but again not available
    with MPI). You can in principle customize what you want to see in your
    animations, but there are few shortcuts:
    
        - :envvar:`animate=1`: complete image, coloured according to effective
          temperature
        - :envvar:`animate=2`: separated images of the (two) components
        - :envvar:`animate='lc'`: image and light curve
        - :envvar:`animate='rv'`: image and radial velocity curve
        - :envvar:`animate='sp'`: image and spectrum
        - :envvar:`animate='pl'`: image and spectrapolarimetry
        - :envvar:`animate='ifm'`: image and IF visibilities
        - :envvar:`animate='am'`: image and astrometry
        
    
    **Example usage**:
    
    First we quickly create a Star like Vega and add multicolour photometry:
    
    >>> vega = phoebe.create.from_library('vega')
    >>> mesh = phoebe.ParameterSet(context='mesh:marching')
    >>> lcobs, lcdeps = phoebe.parse_phot('../../phoebe-testsuite/vega/Vega.phot')
    >>> vega = phoebe.Star(vega, mesh=mesh, pbdep=lcdeps, obs=lcobs)
    
    Then we can easily compute the photometry to match the observations in a few
    alternative ways:
    
    >>> compute(vega)
    >>> compute(vega, subdiv_num=2)
    
    >>> mpi = phoebe.ParameterSet(context='mpi', np=4)
    >>> compute(vega, subdiv_num=2, mpi=mpi)
    
    A shortcut to this function is also available as a method of a Body.
    
    **And then what?**:
    
    Things you might want to do after computing:
    
        - :py:meth:`system.get_synthetic(ref=0, category='lc') <phoebe.backend.universe.Body.get_synthetic>`:
          retrieves the results of a certain category from the the Body.
        - :py:func:`plotting.plot_lcsyn(system, ref=0) <phoebe.backend.plotting.plot_lcsyn>`:
          plots the synthetic to the current axes.
        - :py:func:`plotting.plot_lcobs(system, ref=0) <phoebe.backend.plotting.plot_lcsyn>`:
          plots the observations to the current axes.
        - :py:func:`plotting.plot_lcres(system, ref=0) <phoebe.backend.plotting.plot_lcsyn>`:
          plots the residuals to the current axes.
    
    Or substitute the name of the observations to the ``ref`` keyword. There are
    similar functions or keywords for other observables, simply replace `lc`
    with other types.
    
    @param system: the system to compute
    @type system: Body
    @param params: computational parameterset
    @type params: ParameterSet of context ``compute``.
    @param mpi: parameters describing MPI
    @type mpi: ParameterSet of context 'mpi'
    """
    inside_mpi = kwargs.pop('inside_mpi', None)
    save_result = kwargs.pop('save_result', True)
    
    # Gather the parameters that give us more details on how to compute the
    # system: subdivisions, eclipse detection, optimization flags...
    if extra_func is None:
        extra_func = []
    if extra_func_kwargs is None:
        extra_func_kwargs = [{}]
    
    if params is None:
        params = parameters.ParameterSet(context='compute', **kwargs)
    else:
        params = params.copy()
        for key in kwargs:
            params[key] = kwargs[key]
    auto_detect_circular = True
    
    # Extract info on the time points to compute the system on, and which pbdeps
    # (refs+types) to compute the system for. In principle, we could derive the
    # type from the ref since they are unique, but this way we allow for a
    # possibility to implement 'all lcdep' or so in the future.

    no_mesh_required = extract_times_and_refs(system, params)
    time_per_time = params['time']
    labl_per_time = params['refs']
    type_per_time = params['types']
    samp_per_time = params['samprate']
    
    # Some preprocessing steps
    system.preprocess(time=None)
    
    # Compute stuff for which no mesh is required
    if no_mesh_required:
        for category in no_mesh_required:
            for iref in no_mesh_required[category]:
                itime = no_mesh_required[category][iref]['time']
                isamp = no_mesh_required[category][iref]['samprate']
                getattr(system, category + '_nomesh')(ref=iref, time=itime,
                                    correct_oversampling=isamp,
                                    save_result=save_result)
    
    
    # separate times and refs for datasets that don't need to compute intensities 
    # (ivo = independent_variable_other)
    ivo_time_per_time, ivo_labl_per_time = [], []
    ivo_labl_per_labl, ivo_time_per_labl = [], [] 
    
    # TODO: change this to look in config.indep_var_other instead of hardcoding etvobs
    for i,t in reversed(list(enumerate(time_per_time))): #so pop doesn't change position
        if 'etvobs' in type_per_time[i]:
            if [typ=='etvobs' for typ in type_per_time[i]]==[True]*len(type_per_time[i]):
                # then we want to remove this time stamp from the original list
                ivo_time_per_time.append(time_per_time.pop(i))
                
                labl_per_time_i = labl_per_time.pop(i)
                type_per_time_i = type_per_time.pop(i)
                samp_per_time_i = samp_per_time.pop(i)
                
            else:
                # then there are other types so we just want to copy the time         
                ivo_time_per_time.append(time_per_time[i])

                # reference so pop will remove from original
                labl_per_time_i = labl_per_time[i]
                type_per_time_i = type_per_time[i]
            
            ivo_labl_per_time.append([])
            for j,typ in enumerate(type_per_time_i):
                if typ=='etvobs':
                    labl = labl_per_time_i.pop(j)
                    ivo_labl_per_time[-1].append(labl)
                    
                    if labl not in ivo_labl_per_labl:
                        ivo_labl_per_labl.append(labl)
                        ivo_time_per_labl.append([])
                        k = -1
                    else:
                        k = ivo_labl_per_labl.index(labl)
                        
                    ivo_time_per_labl[k].append(t)
    
    # compute ETVs
    if inside_mpi is None:
        for i,labl in enumerate(ivo_labl_per_labl):
            for body in system.get_bodies():
                body.etv(ref=[labl],times=ivo_time_per_labl[i],ltt=params['ltt'])

    # compute eclipse times in range from min(etv_time_per_time) to
    # max(etv_time_per_time) with some resolution use special effects (apsidal,
    # LTTE, etc) that are enabled in compute options? or etvdep? how do we
    # decide which eclipses to look for without looping through all instances?
    # perhaps we should have been tracking this above since it will be nice to
    # compute whole list once and then pull required times

    # for each time in etv_time_per_time: find correct eclipse time and fill
    # synthetic with etv_time_per_time[i]-computed_time how do we then handle
    # input of cycle? how do we handle changing a period - does that change the
    # calculated eclipse time? do we require input to be cycle after bjd0? and
    # then either plot this as default or derive expected from ephemeris if
    # cycle, how is that defined for secondary eclipses?

    # for all of this - keep in mind how this might work for Nbody case
    # especially enabling/disabling effects which may not be applicable for Nbody

    # should be able to provide ETVs in any of the following formats:
    # cycle et (preferred)
    # cycle etv XXX
    # time et
    # time etv
    # et
    # while providing eclipsing object globally or as a column

    # Some simplifications: try to detect whether a system is circular is not
    system_is_bbag = hasattr(system, 'bodies')
    bbag_has_orbit = system_is_bbag and 'orbit' in system.bodies[0].params              
    
    if system_is_bbag and bbag_has_orbit and auto_detect_circular:
        
        circular = (system.bodies[0].params['orbit']['ecc'] == 0)
        logger.info("Figured out that system is{0}circular".format(circular and ' ' or ' not '))
        
        # Perhaps one of the components is a star with a spot, pulsations or it
        # is misaligned. If so, we can't use the circular optimizations since we
        # need to recompute intensities each iteration.
        for body in system.bodies:
            
            if 'puls' in body.params or 'circ_spot' in body.params:
                circular = False
                logger.info("... but at least one of the components has spots or pulsations: set circular=False")
                break
            elif 'orbit' in body.params and 'theta' in body.params['orbit']: # it is misaligned
                circular = False
                logger.info("... but at least one of the components is misaligned")
                break
            elif hasattr(body,'bodies'):
                circular = False
                logger.info("... but at least one component is a BodyBag")
    else:
        circular = False
        logger.info("Cannot figure out if system is circular or not, leaving at circular={}".format(circular))
    
    # Should we bother with irradiating Bodies and irradiated bodies? There are
    # a few cases here: if the system is circular, we only need to compute the
    # reflection once. In any case, we need to compute bolometric stuff at the
    # desired time stamps (only once for circular cases, otherwise always)
    heating = params['heating'] 
    reflect = params['refl']
    nreflect = params['refl_num']
    ltt = params['ltt']
    beaming = params['beaming_alg']
    mesh_scale = params['mesh_rescale']
    
    # Heating and reflection are by default switched on. However, if there are
    # no irradiators, we don't need to compute it.
    found_irradiator = False
    if heating or reflect:
        for loc, thing in system.walk_all():
            
            if found_irradiator:
                continue
            
            thing_is_parameter = isinstance(thing, parameters.Parameter)
            if thing_is_parameter and thing.get_qualifier() == 'irradiator':
                
                # Now we know 'thing' is the Parameter "irradiator"
                if thing.get_value():
                    
                    # If there's at least one irradiator, we need to leave it on
                    found_irradiator = True
                    logger.info("Including heating and/or reflection effects ({})".format(params['irradiation_alg']))

                    
    # Otherwise we can switch irradiation completely off
    if not found_irradiator or not system_is_bbag:
        heating = False
        reflect = False
    
    # So what about heating then...: if heating is switched on and the orbit is
    # circular, heat only once
    if heating and circular and time_per_time:
        heating = 1
        labl_per_time[0].append('__bol')
        
    # Else heat always
    elif heating and time_per_time:
        for labl in labl_per_time:
            labl.append('__bol')
    
    # and uuhhh... what about reflection? Well, same as for heating: if
    # reflection is switched on, do it only once. Otherwise, reflect always.
    # If heating is not enabled, we need to add the bolometric label.
    if reflect and circular and time_per_time:
        reflect = 1
        if not heating:
            labl_per_time[0].append('__bol')
    elif reflect and not heating and time_per_time:
        for labl in labl_per_time:
            labl.append('__bol')
    
    # Next up: Light time travel effects. Compute for all bodies the conversion
    # between proper time and barycentric time. We need to do two things:
    # first compute the conversions, and then add those arrays to the 'orbsyn'
    # parameterSets, such that 'get_proper_time' knows what to do.
    if ltt and time_per_time:
        # First get a flat list of the bodies
        if hasattr(system, 'get_bodies'):
            bodies = system.get_bodies()
        else:
            bodies = [system]
        
        # Then cycle through all the bodies, and retrieve their full
        # hierarchical orbit
        for body in bodies:
            out = body.get_orbits()
            
            # Once we have the complete orbit for this body, we can compute
            # the proper times from the barycentric times
            objs, vels, prop_times = \
                keplerorbit.get_barycentric_hierarchical_orbit(time_per_time,
                                out[0], out[1])
            
            # Store the results in an "orbsyn" parameterSet
            orbsyn = datasets.DataSet('orbsyn', bary_time=time_per_time,
                                             prop_time=prop_times,
                                             position=objs, velocity=vels,
                                             ref='ltt')
            
            # We need to keep the same hierarchy as with lcsyns and such
            body.params['syn']['orbsyn'] = OrderedDict()
            body.params['syn']['orbsyn'][orbsyn['ref']] = orbsyn
    
    # And don't forget beaming! If the user switches of beaming, we don't
    # care. Else, we check if beaming needs to be computed. Thus, the user can
    # set the beaming algorithm to be switched on, but still disable beaming in
    # some (or all) of the bodies separately.
    beaming_is_relevant = (beaming != 'none')
    if beaming_is_relevant:
        for parset in system.walk():
            if 'beaming' in parset and parset['beaming']:
                beaming_is_relevant = True
                logger.info("Figured out that the system requires beaming")
                break
        else:
            logger.warning(("Beaming algorithm = {} but no beaming Bodies "
                            "found. Check the 'beaming' parameter in the "
                            "Bodies".format(beaming)))
    else:
        logger.info("No beaming included")
        
    # If beaming is not relevant, don't take it into account. Else, if the
    # beaming algorithm is not the "full" one, prepare to store beaming factors
    if not beaming_is_relevant:
        beaming = 'none'
    else:
        system.prepare_beaming(ref='all')
    
    # If we include reflection, we need to reserve space in the mesh for the
    # reflected light. We need to fix the mesh afterwards because each body can
    # have different fields appended in the mesh.
    if system_is_bbag and reflect and len(system)>1:
        system.prepare_reflection(ref='all')
    
    # Scale mesh density
    if mesh_scale != 1:
        for pset in system.walk():
            if pset.get_context() == 'mesh:marching':
                current = marching.delta_to_nelements(pset['delta'])
                pset['delta'] = marching.nelements_to_delta(current*mesh_scale)
        logger.info("{} mesh density with a factor {}".format('Increased' if mesh_scale>1 else 'Decreased', mesh_scale))
    
    # Make sure all Bodies have the same columns in the mesh (we might have
    # added different columns for reflection/beaming for different bodies).
    if system_is_bbag and ((reflect and len(system)>1) or (beaming_is_relevant and not beaming == 'full')):
        system.fix_mesh()
    
    logger.info("Number of subdivision stages: {}".format(params['subdiv_num']))
    
    # We're gonna keep a list of references for which the intensity has been
    # computed already. This can be used for optimization of circular orbits
    system.__had_refs = []
    
    # Now we're ready to do the real stuff
    iterator = zip(time_per_time, labl_per_time, type_per_time, samp_per_time)
    
    # We're gonna compute one time step per turn, but this is different with
    # and without animation.    
    if not animate:
        for i, (time, ref, typ, samp) in enumerate(iterator):
            compute_one_time_step(system, i, time, ref, typ, samp, reflect, nreflect,
                                circular, heating, beaming, params, ltt, save_result,
                                extra_func, extra_func_kwargs)
    
    else:
        if animate is True or animate == 1:
            animate = office.Animation1(system, select='teff')
        elif animate == 2:
            animate = office.Animation4(system, select='proj')
        elif animate == 'lc':
            animate = office.AnimationImLC(system, kwargs1=dict(select='teff'), kwargs2=dict(color='k'))
        elif animate == 'rv':
            animate = office.AnimationImRV(system, kwargs1=dict(select='teff'))
        elif animate == 'sp':
            animate = office.AnimationImSP(system, kwargs1=dict(select='teff'), kwargs2=dict(color='k'))
        
            
        ani = animation.FuncAnimation(pl.gcf(), animate_one_time_step,
                                  range(len(time_per_time)),
                                  fargs=(system, time_per_time, labl_per_time,
                                         type_per_time, samp_per_time,
                                         reflect, nreflect, circular, heating,
                                         beaming, params, ltt, save_result, extra_func,
                                         extra_func_kwargs, animate),
                                  init_func=animate.init_func,
                                  interval=25, repeat=animate.repeat)
        if animate.save:
            ani.save(animate.save[0], **animate.save[1])
        else:
            pl.show()
    
    if inside_mpi is None:
        system.bin_oversampling()
        # We can't compute pblum or l3 inside MPI, because it's this function that
        # is called for different parts of the datasets. So no thread has all the
        # information. This is solved in the MPI decorator, which calls the
        # function after everything is merged.
        system.set_pblum_or_l3()
        #raise SystemExit
        try:
            system.compute_scale_or_offset()
            system.postprocess(time=None)            
        except:
           logger.warning("Cannot compute pblum or l3. I can think of three reasons why this would fail: (1) you're in MPI (2) you have previous results attached to the body (3) you did not give any actual observations, so there is nothing to scale the computations to.")
           raise
    
        # Scale mesh density
        if mesh_scale != 1:
            for pset in system.walk():
                if pset.get_context() == 'mesh:marching':
                    current = marching.delta_to_nelements(pset['delta'])
                    pset['delta'] = marching.nelements_to_delta(current/mesh_scale)
            logger.info("Restored mesh densities")
    
        
        
def observe(system, times, lc=False, rv=False, sp=False, pl=False, mpi=None,
            extra_func=[], extra_func_kwargs=[{}], animate=None,
            save_result=True, **kwargs):
    """
    Customized computation of dependables of a system.
    
    This is similar as :py:func:`compute`. The difference is that in this
    funciton, you have to provide your own times, and also what type of
    observations you want to compute. This is probably useful when you want to
    do simulations of a system, without comparing the observations. It does not
    offer the flexibility to e.g. compute radial velocities and light curves at
    differen times.
    
    This function is equivalent to :py:func:`compute` if you would add ``obs``
    DataSets to the Body where the ``time`` column is everywhere the same.
    
    In contrast to :py:func:`compute`, the parameters to tweak the calculations
    are given as keyword arguments (not as a ParameterSet), and can be any of::
    
            time auto   --   phoebe Compute observables of system at these times
            refs auto   --   phoebe Compute observables of system at these times
           types auto   --   phoebe Compute observables of system at these times
         heating False  --   phoebe Allow irradiators to heat other Bodies
            refl False  --   phoebe Allow irradiated Bodies to reflect light
        refl_num 1      --   phoebe Number of reflections
             ltt False  --   phoebe Correct for light time travel effects
      subdiv_alg edge   --   phoebe Subdivision algorithm
      subdiv_num 3      --   phoebe Number of subdivisions
     eclipse_alg auto   --   phoebe Type of eclipse algorithm

    You can give an optional :ref:`mpi <parlabel-phoebe-mpi>` parameterSet.
    
    Things you might want to do after observing:
    
        - :py:meth:`system.get_synthetic(ref=0, category='lc') <phoebe.backend.universe.Body.get_synthetic>`:
          retrieves the results of a certain category from the the Body.
        - :py:func:`plotting.plot_lcsyn(system, ref=0) <phoebe.backend.plotting.plot_lcsyn>`:
          plots the synthetic to the current axes.
        - :py:func:`plotting.plot_lcobs(system, ref=0) <phoebe.backend.plotting.plot_lcsyn>`:
          plots the observations to the current axes.
        - :py:func:`plotting.plot_lcres(system, ref=0) <phoebe.backend.plotting.plot_lcsyn>`:
          plots the residuals to the current axes.
    
    Or substitute the name of the observations to the ``ref`` keyword. There are
    similar functions or keywords for other observables, simply replace `lc`
    with other types.
    
    """
    if not hasattr(times,'__len__'):
        raise ValueError(("Argument 'times' to observatory.observe needs to be "
                          "a list or an array"))
    # Gather the parameters that give us more details on how to compute the
    # system: subdivisions, eclipse detection, optimization flags...
    params = parameters.ParameterSet(context='compute', **kwargs)
    
    # What do we need to compute? The user only has the possibility of saying
    # e.g. lc=True or lc=['mylc1','mylc2']
    refs = []
    typs = []
    smps = []
    if not lc and not rv and not sp and not pl:
        raise ValueError("You need to compute at least one of lc, rv, sp, pl")
    
    # Derive all lc/rv/...dep parameterset references
    for type in ['lc', 'rv', 'sp', 'pl']:
        if locals()[type] is True:
            if hasattr(system, 'bodies'):
                bodies = system.get_bodies()
            else:
                bodies = [system]
            for body in bodies:
                for parset in body.walk():
                    if parset.context == type+'dep':
                        if parset['ref'] in refs:
                            continue
                        refs.append(parset['ref'])
                        typs.append(type + 'dep')
                        smps.append(1)
    
    # Fill in the parameterSet
    params['time'] = times
    params['refs'] = [refs] * len(times)
    params['types'] = [typs] * len(times)
    params['samprate'] = [smps] * len(times)
    # And run compute
    compute(system, params=params, mpi=mpi, extra_func=extra_func,
            extra_func_kwargs=extra_func_kwargs, animate=animate,
            save_result=save_result)



def binary_eclipse_algorithm(all_systems, algorithm):
    """
    Optimize the eclipse algorithm for binaries.
    
    Predict when there is an eclipse. If there is, use the convex algorithm.
    If there is none, use the "horizon_via_normal" algorithm.
    """
    if algorithm == 'binary':
        # Retrieve the coordinates
        #X1 = all_systems[0].as_point_source(only_coords=True)
        #X2 = all_systems[1].as_point_source(only_coords=True)
        X1 = (all_systems[0].mesh['center']*all_systems[0].mesh['size'][:,None]/all_systems[0].mesh['size'].sum()).sum(axis=0)
        X2 = (all_systems[1].mesh['center']*all_systems[1].mesh['size'][:,None]/all_systems[1].mesh['size'].sum()).sum(axis=0)
            
        # Retrieve the radii of the components        
        d1 = np.sqrt( ((all_systems[0].mesh['center'][:,:2]-X1[:2])**2).sum(axis=1))
        d2 = np.sqrt( ((all_systems[1].mesh['center'][:,:2]-X2[:2])**2).sum(axis=1))
        R1 = np.max(d1)
        R2 = np.max(d2)
        predict_eclipse = np.sqrt( (X1[0]-X2[0])**2 + (X1[1]-X2[1])**2)<=(R1+R2)
        logger.info("{}: predict eclipse (binary Graham E/H)".format(all_systems[0].time))
    else:
        predict_eclipse = None
    
    # If the separation between the two on the sky is smaller or equal to the 
    # sum of the radii, there can be an eclipse.
    if algorithm == 'binary_eclipse' or predict_eclipse:
        
        # If there is a total eclipse, we don't need to do anything either
        # <program this>
        
        # Else, we can't do anything but truly compute the eclipses!
        eclipse.convex_graham([all_systems[0],all_systems[1]])
        return 'binary_eclipse', None
    # Else, we can treat them as simple detached stars
    else:
         eclipse.horizon_via_normal(all_systems)
         return 'binary_separated', None


        
def choose_eclipse_algorithm(all_systems, algorithm='auto'):
    """
    Try to automatically detect the best eclipse detection algorithm.
    
    @param algorithm: override the algorithm in the case there are no eclipses.
    @type algorithm: str, one of 'only_horizon' or 'full'
    """
    found_partial = None
    # Perhaps we need to delegate the eclipse detection algorithm to the
    # bodies in the system: this can happen if we observe two systems
    # simultaneously but are diconnected in reality (e.g. they fall on the same
    # pixels in the CCD, but one is a background, and the other is a foreground
    # object)
    is_bbag = hasattr(all_systems, 'bodies')
    if is_bbag and 'compute' in all_systems.params and not all_systems.connected:
        for system in all_systems.bodies:
            choose_eclipse_algorithm(system, algorithm=compute['algorithm'])
        return compute['algorithm'], found_partial
    elif is_bbag and (not all_systems.connected or len(all_systems)==1):
        for system in all_systems.bodies:
            choose_eclipse_algorithm(system, algorithm=algorithm)
        
        return algorithm, found_partial
    
    # Perhaps we don't want eclipsed
    if algorithm == 'none':
        return algorithm, False
    
    # Perhaps we know there are no eclipses
    if algorithm == 'only_horizon':
        found_partial = eclipse.horizon_via_normal(all_systems)
        return algorithm, found_partial
    
    # Perhaps we know nothing
    elif algorithm == 'full':
        logger.info("Full E/H detection")
        found_partial = eclipse.detect_eclipse_horizon(all_systems)
        return algorithm, found_partial
    
    # Perhaps we know it's a binary
    elif algorithm[:6] == 'binary':
        return binary_eclipse_algorithm(all_systems, algorithm=algorithm)
    
    # Perhaps we want to use the convex one
    elif algorithm == 'convex':
        logger.info("Convex E/H detection")
        try:
            found_partial = eclipse.convex_qhull(all_systems.get_bodies())
        # For single bodies, it's quite trivial:
        except AttributeError:
            found_partial = eclipse.horizon_via_normal(all_systems)
        return algorithm, found_partial
    
    elif algorithm == 'graham':
        logger.info("Graham E/H detection")
               
        try:
            found_partial = eclipse.convex_graham(all_systems.get_bodies())
        # For single bodies, it's quite trivial:
        except AttributeError:
            found_partial = eclipse.horizon_via_normal(all_systems)
        return algorithm, found_partial
    
    # Perhaps we can try to be clever    
    try:
        if hasattr(all_systems,'len') and len(all_systems)==2: # assume it's a binary
            try:
                X1 = all_systems[0].as_point_source(only_coords=True)
                X2 = all_systems[1].as_point_source(only_coords=True)
            except:
                raise ValueError
            
            d1 = np.sqrt( ((all_systems[0].mesh['center'][:,:2]-X1[:2])**2).sum(axis=1))
            d2 = np.sqrt( ((all_systems[1].mesh['center'][:,:2]-X2[:2])**2).sum(axis=1))
            R1 = np.max(d1)
            R2 = np.max(d2)
            if np.sqrt( (X1[0]-X2[0])**2 + (X1[1]-X2[1])**2)<=(R1+R2):
                logger.info("{}: predict eclipse (generic {} E/H)".format(all_systems[0].time,algorithm))
                if algorithm == 'auto':
                    found_partial = eclipse.detect_eclipse_horizon(all_systems)
                elif algorithm=='convex':
                    try:
                        found_partial = eclipse.convex_qhull([all_systems[0],all_systems[1]])
                    except:
                        logger.error("CONVEX THING FAILED AGAIN!!!!")
                        logger.error("Falling back on full detection")
                        found_partial = eclipse.detect_eclipse_horizon(all_systems)
            else:
                logger.info("{}: predict no eclipse (simple E/H)".format(all_systems[0].time))
                found_partial = eclipse.horizon_via_normal(all_systems)
        elif hasattr(all_systems,'len') and len(all_systems)>1: # assume it's a multiple system
            logger.info("{}: predict eclipse (generic E/H)".format(all_systems[0].time))
            if algorithm=='auto':
                found_partial = eclipse.detect_eclipse_horizon(all_systems)
            elif algorithm=='convex':
                try:
                    found_partial = eclipse.convex_qhull(all_systems.get_bodies())
                except:
                    logger.error("CONVEX THING FAILED AGAIN!!!!")
                    logger.error("Falling back on full detection")
                    found_partial = eclipse.detect_eclipse_horizon(all_systems)
            #eclipse.detect_eclipse_horizon(all_systems)
        elif hasattr(all_systems,'params') and 'component' in all_systems.params:
            logger.warning('Perhaps (over)contact system (generic E/H)')
            found_partial = eclipse.detect_eclipse_horizon(all_systems)
        else: # single star, easy detection!
            logger.info("simple E/H detection")
            found_partial = eclipse.horizon_via_normal(all_systems)
    except ValueError:
        logger.info('Could not interpret system, (generic E/H)')
        if algorithm=='auto':
            found_partial = eclipse.detect_eclipse_horizon(all_systems)
        elif algorithm=='convex':
            found_partial = eclipse.convex_qhull(all_systems.get_bodies())
        
    return 'auto', found_partial

#{ Extrafuncs for compute_dependables

def ef_binary_image(system, time, i, name='ef_binary_image',
                    show_orbit=False,
                    axes_on=True, **kwargs):
    """
    Make an image of a binary system.
    
    But setting the x and y limits to sensible values, so that
    we can always see the entire orbit. This eliminates the zoom effect in the
    default behaviour, but of course we need to know the maximum size of the system
    without computing through it. For binary systems, this is of course fairly
    easy.
    """
    kwargs.setdefault('with_partial_as_half', False)
    # Compute the orbit of the system
    if hasattr(system, '__len__'):
        orbit = system[0].params['orbit']
        star1 = system[0]
        star2 = system[1]
    else:
        orbit = system.params['orbit']
        star1 = system
        star2 = None
    period = orbit['period']
    t0,tn = kwargs.pop('t0', 0), kwargs.pop('tn', period)
    times_ = np.linspace(t0, tn, 250)
    orbit1 = keplerorbit.get_binary_orbit(times_, orbit, component='primary')[0]
    orbit2 = keplerorbit.get_binary_orbit(times_, orbit, component='secondary')[0]
    # What's the radius of the stars?
    r1 = coordinates.norm(star1.mesh['_o_center'], axis=1).mean()
    if star2 is not None:
        r2 = coordinates.norm(star2.mesh['_o_center'], axis=1).mean()
    else:
        r2 = r1
    # Compute the limits
    xmin = min(orbit1[0].min(),orbit2[0].min())
    xmax = max(orbit1[0].max(),orbit2[0].max())
    ymin = min(orbit1[1].min(),orbit2[1].min())
    ymax = max(orbit1[1].max(),orbit2[1].max())
    xmin = xmin - 1.1*max(r1,r2)
    xmax = xmax + 1.1*max(r1,r2)
    ymin = ymin - 1.1*max(r1,r2)
    ymax = ymax + 1.1*max(r1,r2)
    # and make the figure
    figsize = kwargs.pop('figsize',8)
    pl.figure(figsize=(figsize,abs(ymin-ymax)/abs(xmin-xmax)*figsize))
    ax = pl.axes([0,0,1,1],aspect='equal',axisbg='k')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    image(system,ax=ax,**kwargs)
    if show_orbit:
        pl.plot(orbit1[0], orbit1[1], 'r-', lw=2)
        pl.plot(orbit2[0], orbit2[1], 'b-', lw=2)
    pl.xlim(xmin,xmax)
    pl.ylim(ymin,ymax)
    if not axes_on:
        ax.set_axis_off()
    pl.savefig('{}_{:04d}'.format(name,i))
    pl.close()


def ef_image(system,time,i,name='ef_image',comp=0,axes_on=True,do_contour=False,
             **kwargs):
    """
    Make an image of a system.
    
    Set ``comp=None`` if you want an image of an entire system, set ``comp=<int>``
    to take an image of one specific component in the system.
    """
    xlim = kwargs.pop('xlim',())
    ylim = kwargs.pop('ylim',())
    # Get the thing to plot
    if hasattr(system,'__len__') and comp is not None:
        system = system[comp]
        savefig = '{}_comp_{:02d}_{:04d}'.format(name, comp, i)
    else:
        savefig = '{}_comp_{:04d}'.format(name, i)
    # and make the figure
    image(system,**kwargs)
    
    if do_contour:
        contour(system, select='longitude', colors='k', linewidths=2, linestyles='-')
        contour(system, select='latitude', colors='k', linewidths=2, linestyles='-')
        
    if not axes_on:
        pl.gca().set_axis_off()
    if xlim:
        pl.xlim(xlim)
    if ylim:
        pl.ylim(ylim)
    pl.savefig(savefig)
    pl.close()
    
    
   

#}
# If this module is run from a terminal as a script, execute the unit tests
if __name__=="__main__":
    import phoebe
    import doctest
    doctest.testmod()





        

    
