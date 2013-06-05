"""
Convert a Body to an observable quantity.

.. autosummary::

    image
    surfmap
    ifm
    make_spectrum
    stokes
    
.. autosummary::
    
    compute
    observe
    
    
"""
import logging
import os
import itertools
import numpy as np
from numpy import pi
from scipy.ndimage.interpolation import rotate as imrotate
from scipy import ndimage
try:
    import pylab as pl
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection,PolyCollection
    import matplotlib as mpl
except ImportError:
    pass
    #print("Soft warning: matplotlib could not be found on your system, 2D plotting is disabled, as well as IFM functionality")
try:
    import pyfits
except ImportError:
    pass
    #print("Unable to load pyfits, you cannot use FITS files")
from phoebe.backend import decorators
from phoebe.utils import plotlib
from phoebe.utils import pergrams
from phoebe.utils import coordinates
from phoebe.utils import utils
from phoebe.units import conversions
from phoebe.units import constants
from phoebe.parameters import parameters
from phoebe.parameters import datasets
from phoebe.algorithms import eclipse
from phoebe.algorithms import reflection
from phoebe.atmospheres import tools
from phoebe.atmospheres import passbands
from phoebe.atmospheres import limbdark
from phoebe.atmospheres import spectra as modspectra
from phoebe.dynamics import keplerorbit

        
np.seterr(all='ignore')
logger = logging.getLogger("OBSERVATORY")
logger.addHandler(logging.NullHandler())


#{ Computing observational quantities
    
def image(the_system,ref='__bol',context='lcdep',
            cmap=None,select='proj',background=None,vmin=None,vmax=None,
            size=800,ax=None,savefig=False,nr=0,
            fourier=False,
            with_partial_as_half=True):
    """
    Compute images of a system or make a 2D plot.
    
    You can make an image from basically any defined observable that has
    fluxes computed. An image is nothing more than a representation of the
    system, using the locally projected flux. To compute a light curve, you
    need this, but also to compute radial velocities, spectra etc... by
    default, however, none of these are used, and the *bolometric* fluxes are
    used. This allows you to make an image of something for which no pbdeps
    are defined. To choose something else, e.g. to make an image in a certain
    passband, you need to specify ``ref`` and ``context``.
    
    All the default parameters are set to make a true flux image of the system,
    in linear grayscale such that white=maximum flux and black=zero flux:
    
    >>> vega = phoebe.create.from_library('vega',create_body=True)
    >>> vega.set_time(0.)
    >>> image(vega)
    
    .. image:: images/backend_observatory_image01.png 
       :scale: 20 %                                   
       :align: center                                 
    
    You can set C{select} to plot effective temperature (C{select='teff'},
    logg (C{select='logg'}) etc instead of projected flux. The colormap is
    adjustable via the C{cmap} keyword, and the background color via keyword
    ``background``. For some selections, there are
    smart default colormaps. E.g. for the effective temperature, the following
    two expressions give the same result:
    
    >>> image(vega,select='teff')
    >>> image(vega,select='teff',cmap=plt.cm.hot,background='0.7')
    
    .. image:: images/backend_observatory_image02.png 
       :scale: 20 %                                   
       :align: center                                 
    
    Setting C{select='rv'} will plot the radial velocity of the system, and then
    the colormap will automatically be changed to ``RdBu``, which means blue for
    velocities towards the observer, white for velocities in the plane of the
    sky, and red for velocities away from the observer. You can adjust the settings
    for the colorscale via ``vmin`` and ``vmax`` (if you set the limits, beware
    that the units of RV are Rsol/d!):
    
    >>> image(vega,select='rv')
    >>> image(vega,select='rv',vmin=-10,vmax=10)
    
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
    C{cmap='blackbody_proj'} and C{cmap='eye'}.
    
    >>> phoebe.image(vega,select='teff',cmap='blackbody')
    >>> phoebe.image(vega,select='teff',cmap='blackbody_proj')
    >>> phoebe.image(vega,select='teff',cmap='blackbody_eye')
    
    +---------------------------------------------------+---------------------------------------------------+---------------------------------------------------+
    | .. image:: images/backend_observatory_image05.png | .. image:: images/backend_observatory_image06.png | .. image:: images/backend_observatory_image07.png |
    |   :width: 233px                                   |   :width: 233px                                   |   :width: 233px                                   |
    |   :align: center                                  |   :align: center                                  |   :align: center                                  |
    +---------------------------------------------------+---------------------------------------------------+---------------------------------------------------+    
    
    By default, a new figure is created, for which axes fill the whole canvas
    and the x and y-axis are completely removed. If you want to add an image
    to an existing axis, that is possible by giving that axis as an argument.
    In this case, the limits are not automatically set, so you need to set
    them manually. Luckily, this function returns a recommendation for the
    limits, as well as the collection of triangels themselves. The latter can
    be helpful if you want to add a colorbar. These options provide with you
    with the utmost flexibility to incorporate the image of your Body in
    whatever customized plot or subplot you want.
    
    >>> xlim,ylim,patch = phoebe.image(vega,ax=ax,background='white')
    >>> plt.xlim(xlim)
    >>> plt.ylim(ylim)
    >>> plt.xlabel("X-Distance [$R_\odot$]")
    >>> plt.ylabel("Y-Distance [$R_\odot$]")
    
    >>> cbar = plt.colorbar(patch)
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
    time image. This allows you to create and save an image of a Body with one
    single command.
    
    Finally, an experimental option is to compute the Fourier transform of
    an image instead of the normal image:
    
    >>> image(vega,fourier=True)
    
    .. image:: images/backend_observatory_image10.png 
       :width: 233px                                   
       :align: center                                 
    
    @return: x limits, y limits, patch collection
    @rtype: tuple, tuple, patch collection
    """
    #-- default color maps and background depend on the type of dependables:
    if cmap is None and select=='rv':
        cmap = pl.cm.RdBu_r
    elif cmap is None and select=='teff':
        cmap = pl.cm.hot
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
    if fourier and ax is None:
        fig = pl.figure(figsize=(3,3))
    elif ax is None:
        fig = pl.figure(figsize=(size/100.,size/100.))
    if ax is None:
        #-- the axes should be as big as the figure, so that there are no margins
        ax = pl.axes([0,0,1,1],axisbg=background,aspect='equal')
        fig = pl.gcf()
        #-- make sure the background and outer edge of the image are black
        fig.set_facecolor(background)
        fig.set_edgecolor(background)
        axis_created = True
    else:
        ax.set_axis_bgcolor(background)
        ax.set_aspect('equal')
        axis_created = False
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
    if not cmap_ in ['blackbody_proj','eye']:
        p = PolyCollection(mesh['triangle'].reshape((-1,3,3))[:,:,:2],
                             array=values,
                             closed=True,
                             edgecolors=cmap(colors),
                             facecolors=cmap(colors),
                             cmap=cmap)
    elif cmap_=='blackbody_proj':
        values = np.abs(mesh['proj_'+ref]/mesh['mu'])
        if 'refl_'+ref in mesh.dtype.names:
            values += mesh['refl_'+ref]
        values = (values/values.max()).reshape((-1,1))*np.ones((len(values),4))
        values[:,-1] = 1.
        colors = np.array([cmap(c) for c in colors])*values 
        
        p = PolyCollection(mesh['triangle'].reshape((-1,3,3))[:,:,:2],
                             closed=True,
                             edgecolors=colors,
                             facecolors=colors)
        
        
        #for i,triangle in enumerate(mesh['triangle']):
            #patches.append(Polygon(triangle.reshape((3,3))[:,:2],closed=True,edgecolor=tuple(colors[i])))
        #p = PatchCollection(patches,cmap=cmap)
        ##-- set the face colors of the triangle plot objects, and make sure
        ##   the edges have the same color
        #p.set_edgecolor([tuple(c) for c in colors])
        #p.set_facecolor([tuple(c) for c in colors])
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
        
        p = PolyCollection(mesh['triangle'].reshape((-1,3,3))[:,:,:2],
                             closed=True,
                             edgecolors=colors,
                             facecolors=colors)
        #for i,triangle in enumerate(mesh['triangle']):
            #patches.append(Polygon(triangle.reshape((3,3))[:,:2],closed=True,edgecolor=tuple(colors[i])))
        #p = PatchCollection(patches,cmap=cmap)
        ##-- set the face colors of the triangle plot objects, and make sure
        ##   the edges have the same color
        #p.set_edgecolor([tuple(c) for c in colors])
        #p.set_facecolor([tuple(c) for c in colors])
    #-- set the color scale limits
    if vmin is not None: vmin_ = vmin
    if vmax is not None: vmax_ = vmax
    
    p.set_clim(vmin=vmin_,vmax=vmax_)    
    #-- add the triangle plot objects to the axis, and set the axis limits to
    #   be a tiny bit larger than the object we want to plot.
    ax.add_collection(p)
    #-- derive the limits for the axis
    # old style:
    #lim_min = min(x.min(),y.min())-0.01*x.ptp()
    #lim_max = max(x.max(),y.max())+0.01*x.ptp()
    #pl.xlim(lim_min,lim_max)
    #pl.ylim(lim_min,lim_max)
    # new style:
    #-- dont be smart when axis where given
    offset_x = (x.min()+x.max())/2.0
    offset_y = (y.min()+y.max())/2.0
    margin = 0.01*x.ptp()
    lim_max = max(x.max()-x.min(),y.max()-y.min())
    ax.set_xlim(offset_x-margin-lim_max/2.0,offset_x+lim_max/2.0+margin)
    ax.set_ylim(offset_y-margin-lim_max/2.0,offset_y+lim_max/2.0+margin)
    if axis_created:
        ax.set_xticks([])
        ax.set_yticks([])
        #pl.box(on=False)
    
    #-- compute Fourier transform of image
    if fourier:
        #-- save the above image, so that we can read it in to compute
        #   the 2D Fourier transform.
        pl.savefig('__temp.png',facecolor='k',edgecolor='k');pl.close()
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
    
    xlim,ylim = ax.get_xlim(),ax.get_ylim()
    if savefig==True:
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
        d = the_system.as_point_source()['coordinates'][2]
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
    
    return xlim,ylim,p


def surfmap(the_system,ref='__bol',context='lcdep',cut=0.96,
            cmap=None,select='proj',background=None,vmin=None,vmax=None,
            size=800,ax=None,savefig=False,nr=0,with_partial_as_half=True):
    """
    Compute images of a system or make a 2D plot.
    
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
        cmap = pl.cm.hot
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


    
    
def ifm(the_system,posangle=0.0,baseline=0.0,eff_wave=None,ref=0,
        figname=None,keepfig=True):
    """
    Compute the Fourier transform of the system along a baseline.
    
    If you give a single positional angle as a float instead of an array,
    the whole profile will be computed. Otherwise, only selected baselines
    will be computed, to e.g. match observed data.
    
    xlims and ylims in Rsol.
    
    .. note::
    
        Thanks to M. Hillen. He is not responsible for bugs or errors.
    """
    #-- information on what to compute
    data_pars,ref = the_system.get_parset(ref)
    passband = data_pars.request_value('passband')
    
    #-- helper function
    def rotmatrix(theta): return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    
    #-- make sure we can cycle over the baselines and posangles
    single_baseline = False
    if not hasattr(baseline,'__iter__'):
        single_baseline = True
        baseline = [baseline]
    if not hasattr(posangle,'__iter__'):
        posangle = [posangle]
    
    #-- prepare output
    frequency_out = []
    baseline_out = []
    posangle_out = []
    visibility_out = []
    phase_out = []
    angular_scale_out = []
    angular_profile_out = []
    
    #-- set effective wavelength to the one from the passband if not given
    #   otherwise
    if eff_wave is None:
        eff_wave = passbands.get_info([passband])['eff_wave'][0]*np.ones(len(posangle))
    logger.info("ifm: cyclic frequency to m at lambda~%.4g angstrom"%(eff_wave.mean()))
    #-- make an image if necessary, but in any case retrieve it's
    #   dimensions
    if figname is None:
        figname = 'ifmfig_temp.png'
        keep_figname = False
        xlims,ylims,p = image(the_system,ref=ref,savefig=figname)
    else:
        keep_figname = True
        figname,xlims,ylims = figname
    coords =np.array([[i,j] for i,j in itertools.product(xlims,ylims)])
    #-- load the image as a numpy array
    data = pl.imread(figname)[:,:,0]
    if not keep_figname:
        os.unlink(figname)
    #-- cycle over all baselines and position angles
    d = the_system.as_point_source()['coordinates'][2]#list(the_system.params.values())[0].request_value('distance','Rsol')
    dpc = conversions.convert('Rsol','pc',d)
        
    for nr,(bl,pa,wl) in enumerate(zip(baseline,posangle,eff_wave)):        
        
        if keepfig:
            xlims,ylims,p = image(the_system,ref=ref,savefig='{}_{:05d}.fits'.format(keepfig,nr))
            xlims,ylims,p = image(the_system,ref=ref)
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
            resol = conversions.convert('rad','mas',resol)/npix
            pl.annotate('PA={:.2f}$^\circ$\n$\lambda$={:.0f}$\AA$\nB={:.0f}m\nU,V=({:.1f},{:.1f}) m\n{:d} pix\n{:.3g} mas/pix'.format(pa,eff_wave,bl,uc,vc,npix,resol),(0.95,0.95),va='top',ha='right',xycoords='axes fraction',color='r',size=20)
        #-- rotate counter clockwise by angle in degrees, and recalculate the
        #   values of the corners
        data_ = imrotate(data,-pa,reshape=True,cval=0.) # was -pa
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
        x = conversions.convert('rad','as',x) # arseconds
        x -= x[0]
        if single_baseline and bl==0: 
            nyquist = 0.5/x[1]
            f0 = 0.01/x.ptp()
            fn = nyquist/25.
            df = 0.01/x.ptp()
            logger.info('ifm: single baseline equal to 0m: computation of entire profile')
        else:
            f0 = conversions.convert('m','cy/arcsec',bl,wave=(wl,'angstrom'))*2*np.pi
            fn = f0
            df = 0
            logger.info('ifm: computation of frequency and phase at f0={:.3g} cy/as (lam={:.3g}AA)'.format(f0,wl))
            if keepfig and keepfig is not True:
                pl.annotate('f={:.3g} cy/as\n d={:.3g} pc'.format(f0,dpc),(0.95,0.05),va='bottom',ha='right',xycoords='axes fraction',color='r',size=20)
                pl.figure()
                pl.plot(x*1000.,signal,'k-')
                pl.grid()
                pl.xlabel('Coordinate [mas]')
                pl.ylabel("Flux")
                pl.savefig('{}_{:05d}_prof.png'.format(keepfig,nr))
                pl.close()
        #-- to take band pass smearing into account, we need to let the
        #   wavelength vary over the passband, and add up all the
        #   Fourier transforms but weighted with the SED intensity
        #   times the transmission filter.
        signal = signal/signal.sum()
        f1,s1 = pergrams.DFTpower(x,signal,full_output=True,
                            f0=f0,fn=fn,df=df)
        s1_vis = np.abs(s1)
        s1_phs = np.angle(s1)
        #-- correct cumulative phase
        s1_phs = s1_phs - x.ptp()*pi*f1
        s1_phs = (s1_phs % (2*pi))# -pi
        b1 = conversions.convert('cy/arcsec','m',f1,wave=(wl,'angstrom'))
        b1/=(2*np.pi)
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
        
    if single_baseline and baseline[0]==0:
        frequency_out = frequency_out[0]
        baseline_out = baseline_out[0]
        posangle_out = posangle_out[0]
        visibility_out = visibility_out[0]
        phase_out = phase_out[0]
        angular_scale_out = angular_scale_out[0]
        angular_profile_out = angular_profile_out[0]
    else:
        frequency_out = np.array(frequency_out)
        baseline_out = np.array(baseline_out)
        posangle_out = np.array(posangle_out)
        visibility_out = np.array(visibility_out)
        phase_out = np.array(phase_out)
        #angular_scale_out = np.array(angular_scale_out)
        #angular_profile_out = np.array(angular_profile_out)
    
    return frequency_out,baseline_out,posangle_out,\
           visibility_out,phase_out,\
           angular_scale_out,angular_profile_out

            
def make_spectrum(the_system,wavelengths=None,sigma=2.,depth=0.4,ref=0,rv_grav=True):
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
    wavelength range (in fact the user is responsible for this). Limb darkening
    coefficients are thus not wavelength dependent within one spectrum or
    spectral line profile, in the current implementation.
    
    Returns: wavelength ranges, total (unnormalized) spectrum, total continuum.
    
    You can get the normalised spectrum by dividing the flux with the
    continuum.
    
    @param wavelengths: predefined array or list with 3 elements (central wavelength (angstrom), range in km/s and number of wavelength points.
    @param sigma: intrinsic width of the Gaussian profile in km/s
    @type sigma: float
    """
    #-- we need some information on how to compute the spectrum that is not
    #   available in the pbdep parameterSet: we need to know which wavelength
    #   range, and how to sample that wavelength range. We seek this
    #   information first in the observations parameterSet, if there is one.
    #   if there are none, we look for it in the synthetic parameterSet.
    #   The same holds for information on a posteriori inclusion of
    #   macroturbulence, and instrumental broadening
    #-- is there data available? then we can steal the wavelength array
    #   from there. Otherwise we turn to the synthetic parameterSet.
    clip_after = None
    iobs,refo_ = the_system.get_parset(ref=ref,context='spobs')
    isyn,refs_ = the_system.get_parset(ref=ref,context='spsyn')
    wc = None
    if refo_ is not None:
        if not 'wavelength' in iobs or not len(iobs['wavelength']):
            iobs.load()
        wavelengths = iobs['wavelength']
        R = iobs['R']
        if 'vgamma' in iobs:
            vgamma = iobs.get_value('vgamma','km/s')
            wavelengths = tools.doppler_shift(wavelengths,vgamma)
        if 'vmacro' in iobs:
            vmacro = iobs['vmacro']
        if 'clambda' in iobs:
            wc = iobs.get_value('clambda','AA')
        #-- make wavelength range a little bit broader and clip afterwards
        #   we do this to take into account neighbouring lines
        clip_after = wavelengths[0],wavelengths[-1]
        dw = wavelengths[1]-wavelengths[0]
        wavelengths = np.hstack([np.arange(wavelengths[0]-10.,wavelengths[0],dw),
                                 wavelengths,
                                 np.arange(wavelengths[-1],wavelengths[-1]+10,dw)+dw])
    else:
        R = None
        vmacro = 0.
    #-- information on dependable set
    idep,ref = the_system.get_parset(ref=ref,context='spdep')
    ld_model = idep['ld_func']
    method = idep['method']
    keep = the_system.mesh['mu']<=0
    #-- create the wavelength array from the information of the spdep: we use
    #   central wavelength, velocity range and spectral resolving power.
    #   We broaden the range of the wavelengths a bit, so that we are sure that
    #   also neighbouring lines are taken into account. We clip the spectrum
    #   afterwards if needed. If you give wavelengths yourself, you need to
    #   take this into account yourself.

    logger.info('Computing spectrum of {}'.format(ref))
    if wavelengths is None:
        wc = idep.get_value('clambda','AA')
        vl = idep.get_value_with_unit('max_velo')
        w0 = conversions.convert(vl[1],'AA',-2*vl[0],wave=(wc,'AA'))
        wn = conversions.convert(vl[1],'AA',+2*vl[0],wave=(wc,'AA'))
        Npoints = (wn-w0) / (wc/idep['R'])
        logger.info('Created wavelength array of length {} between {}AA and {}AA - clipped afterwards'.format(Npoints,w0,wn))
        wavelengths = np.linspace(w0,wn,Npoints)
        clip_after = conversions.convert(vl[1],'AA',-vl[0],wave=(wc,'AA')),\
                     conversions.convert(vl[1],'AA',+vl[0],wave=(wc,'AA'))
    #-- for convenience, the user might give the wavelength array at this level
    #   to using central wavelength, velocity range and number of points.
    elif len(wavelengths)==3:
        w0 = conversions.convert('km/s','AA',-wavelengths[1],wave=(wavelengths[0],'AA'))
        wn = conversions.convert('km/s','AA',+wavelengths[1],wave=(wavelengths[0],'AA'))
        wavelengths = np.linspace(w0,wn,wavelengths[2])
    #-- else, we assume wavelengths is already an array, so we don't need to do
    #   anything, except for setting the central wavelength "wc".
    elif wc is None:
        wc = (wavelengths[0]+wavelengths[-1])/2.    
    #-- if we're not seeing the star, we can easily compute the spectrum: it's
    #   zero!
    if not np.sum(keep):
        logger.info('Still need to compute (projected) intensity')
        the_system.intensity(ref=ref)
    the_system.projected_intensity(ref=ref,method='numerical')
    keep = the_system.mesh['proj_'+ref]>0
    if not np.sum(keep):
        logger.info('no spectrum synthesized (nothing visible), zero flux received')
        return wavelengths,np.zeros(len(wavelengths)),0.
    
    if method=='numerical' and not idep['profile']=='gauss':    
        #-- get limb angles
        mus = the_system.mesh['mu']
        keep = (mus>0) & (the_system.mesh['partial'] | the_system.mesh['visible'])# & -np.isnan(self.mesh['size'])
        mus = mus[keep]
        #-- negating the next array gives the partially visible things, that is
        #   the only reason for defining it.
        visible = the_system.mesh['visible'][keep]
        #-- compute normalised intensity using the already calculated limb darkening
        #   coefficents. They are normalised so that center of disk equals 1. Then
        #   these values are used to compute the weighted sum of the spectra.
        logger.info('using limbdarkening law {} - spectra interpolated from grid {}'.format(ld_model,idep['profile']))
        Imu = getattr(limbdark,'ld_%s'%(ld_model))(mus,the_system.mesh['ld_'+ref][keep].T)
        teff,logg = the_system.mesh['teff'][keep],the_system.mesh['logg'][keep]
        #-- fitters can go outside of the grid
        try:
            spectra = modspectra.interp_spectable(idep['profile'],teff,logg,wavelengths)
        except IndexError:
            logger.error('no spectrum synthesized (outside of grid ({}<=teff<={}, {}<=logg<={}), zero flux received'.format(teff.min(),teff.max(),logg.min(),logg.max()))
            return wavelengths,np.zeros(len(wavelengths)),0.
    
        proj_intens = spectra[1]*mus*Imu*the_system.mesh['size'][keep]
        rad_velos = -the_system.mesh['velo___bol_'][keep,2]
        rad_velos = conversions.convert('Rsol/d','km/s',rad_velos)
        if hasattr(the_system,'params') and 'vgamma' in the_system.params.values()[0]:
            vgamma = the_system.params.values()[0].get_value('vgamma','km/s')
            rad_velos += vgamma
            logger.info('Systemic radial velocity = {:.3f} km/s'.format(vgamma))
        logger.info('synthesizing spectrum using %d faces (RV range = %.6g to %.6g km/s)'%(len(proj_intens),rad_velos.min(),rad_velos.max()))

        total_continum = 0.
        total_spectrum = 0.
        #-- gravitational redshift:
        if rv_grav:
            rv_grav = 0#generic.gravitational_redshift(the_system)
        for i,rv in enumerate(rad_velos):
            total_spectrum += tools.doppler_shift(wavelengths,rv+rv_grav,flux=spectra[0,:,i]*proj_intens[:,i])
            total_continum += tools.doppler_shift(wavelengths,rv+rv_grav,flux=proj_intens[:,i])
    elif method=='numerical':
        #-- derive intrinsic width of the profile
        sigma = conversions.convert('km/s','AA',sigma,wave=(wc,'AA'))-wc
        logger.info('Intrinsic width of the profile: {} AA'.format(sigma))
        template = 1.00 - depth*np.exp( -(wavelengths-wc)**2/(2*sigma**2))
        proj_intens = the_system.mesh['proj_'+ref][keep]
        rad_velos = -the_system.mesh['velo___bol_'][keep,2]
        rad_velos = conversions.convert('Rsol/d','km/s',rad_velos)
        if hasattr(the_system,'params') and 'vgamma' in the_system.params.values()[0]:
            vgamma = the_system.params.values()[0].get_value('vgamma','km/s')
            rad_velos += vgamma
            logger.info('Systemic radial velocity = {:.3f} km/s'.format(vgamma))
        sizes = the_system.mesh['size'][keep]
        logger.info('synthesizing Gaussian profile using %d faces (RV range = %.6g to %.6g km/s)'%(len(proj_intens),rad_velos.min(),rad_velos.max()))
        total_continum = np.zeros_like(wavelengths)
        total_spectrum = 0.
        #-- gravitational redshift:
        if rv_grav:
            rv_grav = 0#generic.gravitational_redshift(the_system)
        for pri,rv,sz in zip(proj_intens,rad_velos,sizes):
            spec = pri*sz*tools.doppler_shift(wavelengths,rv+rv_grav,flux=template)
            total_spectrum += spec
            total_continum += pri*sz
    
    elif method=='analytical' and idep['profile']=='gauss':
        #-- For the analytical computation, we require a linear limb darkening
        #   law
        if not idep['ld_func']=='linear':
            raise ValueError("Analytical computation of spectrum requires a 'linear' limb-darkening model (not '{}')".format(idep['ld_func']))
        epsilon = idep['ld_coeffs'][0]
        vrot = the_system.mesh['velo___bol_'][:,2].max() # this is vsini!
        vrot = conversions.convert('Rsol/d','km/s',vrot)
        logger.info('analytical rotational broadening with veq=%.6f km/s'%(vrot))
        #teff = the_system.params['star']['teff']
        #logg = the_system.params['star'].request_value('logg','[cm/s2]')
        #spectra = limbdark.interp_spectable('atlas',[teff],[logg],wavelengths)
        #spectrum = spectra[0]/spectra[1]
        sigma = conversions.convert('km/s','AA',sigma,wave=(wc,'AA'))-wc
        logger.info('Intrinsic width of the profile: {} AA'.format(sigma))
        template = 1.00 - depth*np.exp( -(wavelengths-wc)**2/(2*sigma**2))
        wavelengths,total_spectrum = tools.rotational_broadening(wavelengths,template,vrot,stepr=-1,epsilon=epsilon)
        total_continum = np.ones_like(wavelengths)
   
    #-- convolve with instrumental profile if desired
    if R is not None:
        instr_fwhm = wc/R
        instr_sigm = instr_fwhm/2.38
        logger.info('Convolving spectrum with instrumental profile of FWHM={:.3f}AA'.format(instr_fwhm))
        wavelengths,total_spectrum = tools.rotational_broadening(wavelengths,
                       total_spectrum/total_continum,vrot=0.,vmac=vmacro,fwhm=instr_sigm)
        total_spectrum *= total_continum
   
    if clip_after is not None:
        keep = (clip_after[0]<=wavelengths) & (wavelengths<=clip_after[1])
        wavelengths = wavelengths[keep]
        total_spectrum = total_spectrum[keep]
        total_continum = total_continum[keep]
        
    
    
        
        
    return wavelengths,total_spectrum,total_continum

def stokes(the_system,wavelengths=None,sigma=2.,depth=0.4,ref=0,
           rv_grav=True,do_V=True,do_Q=False,do_U=False):
    r"""
    Calculate the stokes profiles.
    
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
    
    mesh = the_system.mesh
    #-- is there data available? then we can steal the wavelength array
    #   from there. We also collect information on the resolution and on the
    #   macro turbulent velocity. It might seem strange to derive vmacro from
    #   the observation set, but that's because in this approximation it isn't
    #   real part of the model, but is rather done after computing the whole
    #   spectrum.
    #   Finally, if data is found, we check what to compute from the
    #   observation set.
    iobs,ref_ = the_system.get_parset(ref=ref,context='plobs')
    if ref_ is not None:
        if not 'wavelength' in iobs or not len(iobs['wavelength']):
            iobs.load()
        wavelengths = iobs.request_value('wavelength','AA')
        #-- instrumental info
        R = iobs['R']
        if 'vmacro' in iobs:
            vmacro = iobs['vmacro']
        #-- profiles to compute, and method to use
        if 'V' in iobs['columns']:
            do_V = True
        if 'Q' in iobs['columns']:
            do_Q = True
        if 'U' in iobs['columns']:
            do_U = True
        
    else:
        R = None
        vmacro = 0.
    #-- information on dependable set: we need the limb darkening function,
    #   the method, the glande factor and the weak field approximation boolean
    idep,ref = the_system.get_parset(ref=ref,context='pldep')
    ld_model = idep['ld_func']
    method = idep['method']
    glande = idep['glande']
    weak_field = idep['weak_field']
    keep = the_system.mesh['mu']<=0
    #-- create the wavelength array from the information of the spdep: we use
    #   central wavelength, velocity range and spectral resolving power.
    #   We broaden the range of the wavelengths a bit, so that we are sure that
    #   also neighbouring lines are taken into account. We clip the spectrum
    #   afterwards if needed. If you give wavelengths yourself, you need to
    #   take this into account yourself.
    clip_after = None
    logger.info('Computing spectropolarimetry of {}'.format(ref))
    if wavelengths is None:
        wc = idep.get_value('clambda','AA')
        vl = idep.get_value_with_unit('max_velo')
        w0 = conversions.convert(vl[1],'AA',-2*vl[0],wave=(wc,'AA'))
        wn = conversions.convert(vl[1],'AA',+2*vl[0],wave=(wc,'AA'))
        Npoints = (wn-w0) / (wc/idep['R'])
        logger.info('Created wavelength array of length {} between {}AA and {}AA - clipped afterwards'.format(Npoints,w0,wn))
        wavelengths = np.linspace(w0,wn,Npoints)
        clip_after = conversions.convert(vl[1],'AA',-vl[0],wave=(wc,'AA')),\
                     conversions.convert(vl[1],'AA',+vl[0],wave=(wc,'AA'))
    #-- for convenience, the user might give the wavelength array at this level
    #   to using central wavelength, velocity range and number of points.
    elif len(wavelengths)==3:
        w0 = conversions.convert('km/s','AA',-wavelengths[1],wave=(wavelengths[0],'AA'))
        wn = conversions.convert('km/s','AA',+wavelengths[1],wave=(wavelengths[0],'AA'))
        wavelengths = np.linspace(w0,wn,wavelengths[2])
    #-- else, we assume wavelengths is already an array, so we don't need to do
    #   anything, except for setting the central wavelength "wc".
    else:
        wc = (wavelengths[0]+wavelengths[-1])/2.
    #-- if we're not seeing the star, we can easily compute the spectrum: it's
    #   zero! Muhuhahaha!
    if not np.sum(keep):
        logger.info('Still need to compute (projected) intensity')
        the_system.intensity(ref=ref)
    the_system.projected_intensity(ref=ref,method='numerical')
    keep = the_system.mesh['proj_'+ref]>0
    if not np.sum(keep):
        logger.info('no spectropolarimetry synthesized, zero flux received')
        return wavelengths,np.zeros(len(wavelengths)),np.zeros(len(wavelengths)),\
               np.zeros(len(wavelengths)),np.zeros(len(wavelengths)),np.ones(len(wavelengths))
    
    #-- magnitude of magnetic field and angle towards the LOS
    B = coordinates.norm(mesh['B_'][keep],axis=1)*1e-4 # and convert to Tesla 
    cos_theta = coordinates.cos_angle(mesh['B_'][keep],np.array([[0,0.,-1]]),axis=1)
    sin2theta = 1 - cos_theta**2
    cos_chi = coordinates.cos_angle(mesh['B_'][keep],np.array([[1.,0.,0]]),axis=1)
    cos2chi = cos_chi**2
    sin2chi = 1. - cos2chi
    cos22chi = (cos2chi - sin2chi)**2
    sin22chi = 1. - cos22chi
    #-- Zeeman splitting in angstrom
    #bohr_magneton = 9.27400968e-24
    #delta_nu_zeemans = -glande*bohr_magneton*B
    delta_nu_zeemans = -glande*constants.qe*B/(4*np.pi*constants.me) # (qe is negative but somewhere my B field has wrong sign?)
    delta_nu_zeemans2 = delta_nu_zeemans**2
    delta_v_zeemans = (wc*1e-10)*delta_nu_zeemans/1000. # from cy/s to km/s
    #-- radial velocities
    rad_velos = -the_system.mesh['velo___bol_'][keep,2]
    rad_velos = conversions.convert('Rsol/d','km/s',rad_velos)    
    nus = conversions.convert('AA','Hz',wavelengths)    
    logger.info('{} approximation'.format(weak_field and 'Weak-field' or 'No weak-field'))
    if method=='numerical' and not idep['profile']=='gauss':
        #-- get limb angles
        mus = the_system.mesh['mu']
        keep = (mus>0) & (the_system.mesh['partial'] | the_system.mesh['visible'])# & -np.isnan(self.mesh['size'])
        mus = mus[keep]
        #-- negating the next array gives the partially visible things, that is
        #   the only reason for defining it.
        visible = the_system.mesh['visible'][keep]
        #-- compute normalised intensity using the already calculated limb darkening
        #   coefficents. They are normalised so that center of disk equals 1. Then
        #   these values are used to compute the weighted sum of the spectra.
        logger.info('using limbdarkening law {} - spectropolarimetry interpolated from grid {}'.format(ld_model,idep['profile']))
        Imu = getattr(limbdark,'ld_%s'%(ld_model))(mus,the_system.mesh['ld_'+ref][keep].T)
        teff,logg = the_system.mesh['teff'][keep],the_system.mesh['logg'][keep]
        #-- fitters can go outside of the grid
        try:
            spectra = modspectra.interp_spectable(idep['profile'],teff,logg,wavelengths)
        except IndexError:
            logger.error('no spectropolarimetry synthesized (outside of grid ({}<=teff<={}, {}<=logg<={}), zero flux received'.format(teff.min(),teff.max(),logg.min(),logg.max()))
            return wavelengths,np.zeros(len(wavelengths)),0.
    
        proj_intens = spectra[1]*mus*Imu*the_system.mesh['size'][keep]
        logger.info('synthesizing spectropolarimetry using %d faces (RV range = %.6g to %.6g km/s)'%(len(proj_intens),rad_velos.min(),rad_velos.max()))

        total_continum = 0.
        stokes_I = 0.
        stokes_V = 0.
        #-- gravitational redshift:
        if rv_grav:
            rv_grav = 0#generic.gravitational_redshift(the_system)
        for i,rv in enumerate(rad_velos):
            rvz = delta_v_zeemans[i]
            total_continum += tools.doppler_shift(wavelengths,rv+rv_grav,flux=proj_intens[:,i])
            
            specm = tools.doppler_shift(wavelengths,rv+rv_grav-rvz,flux=spectra[0,:,i]*proj_intens[:,i])
            specp = tools.doppler_shift(wavelengths,rv+rv_grav+rvz,flux=spectra[0,:,i]*proj_intens[:,i])
            
            stokes_I += tools.doppler_shift(wavelengths,rv+rv_grav,flux=spectra[0,:,i]*proj_intens[:,i])
            
            #-- Stokes V in weak field approximation or not
            if do_V and weak_field:
                stokes_V -= costh*delta_nu_zeemans[i]*utils.deriv(nus,spec)
            elif do_V:
                stokes_V += cos_theta[i]*(specm-specp)/2.
    elif method=='numerical':
        #-- derive intrinsic width of the profile
        sigma = conversions.convert('km/s','AA',sigma,wave=(wc,'AA'))-wc
        logger.info('Intrinsic width of the profile: {} AA'.format(sigma))
        template = 1.00 - depth*np.exp( -(wavelengths-wc)**2/(2*sigma**2))
        proj_intens = the_system.mesh['proj_'+ref][keep]
        sizes = the_system.mesh['size'][keep]
        logger.info('synthesizing Gaussian profile using %d faces (sig= %.2e AA,RV range = %.6g to %.6g km/s)'%(len(proj_intens),sigma,rad_velos.min(),rad_velos.max()))
        total_continum = np.zeros_like(wavelengths)
        stokes_I = 0.
        stokes_V = 0.
        stokes_Q = 0.
        stokes_U = 0.
        #-- gravitational redshift:
        if rv_grav:
            rv_grav = 0#generic.gravitational_redshift(the_system)
        rad_velosw = conversions.convert('km/s','AA',rad_velos,wave=(wc,'AA'))-wc
        for i,(pri,rv,sz,iB,costh) in enumerate(zip(proj_intens,rad_velos,sizes,B,cos_theta)):
            rvz = delta_v_zeemans[i]
            #-- first version
            spec  = pri*sz*tools.doppler_shift(wavelengths,rv+rv_grav,flux=template)
            specm = pri*sz*tools.doppler_shift(wavelengths,rv+rv_grav-rvz,flux=template)
            specp = pri*sz*tools.doppler_shift(wavelengths,rv+rv_grav+rvz,flux=template)
            
            #-- we can compute Stokes V in weak field approximation or not
            if do_V and weak_field:
                stokes_V -= costh*delta_nu_zeemans[i]*utils.deriv(nus,spec)
            elif do_V:
                stokes_V += costh*(specm-specp)/2.
                
            #-- second version: weak field approximation
            #mytemplate = pri*sz*(1.00 - depth*np.exp( -(wavelengths-wc-rad_velosw[i])**2/(2*sigma**2)))
            #stokes_V -= costh*delta_nu_zeemans[i]*utils.deriv(nus,mytemplate)
            #- third version: weak field approximation
            #stokes_V -= costh*delta_nu_zeemans[i]*utils.deriv(nus,spec)
            
            #-- Stokes Q and U: this must be in weak field approximation for now
            sec_deriv = utils.deriv(nus,utils.deriv(nus,spec))
            if do_Q:
                stokes_Q -= 0.25*sin2th*cos22chi*delta_nu_zeemans2[i]*sec_deriv
            if do_U:
                stokes_U -= 0.25*sin2th*cos22chi*delta_nu_zeemans2[i]*sec_deriv
            
            stokes_I += spec
            total_continum += pri*sz
        #logger.info("Zeeman splitting: between {} and {} AA".format(min(conversions.convert('Hz','AA',delta_nu_zeemans,wave=(wc,'AA'))),max(conversions.convert('Hz','AA',delta_nu_zeemans,wave=(wc,'AA')))))
        #logger.info("Zeeman splitting: between {} and {} Hz".format(min(delta_nu_zeemans/1e6),max(delta_nu_zeemans/1e6)))
        logger.info("Zeeman splitting: between {} and {} km/s".format(min(delta_v_zeemans),max(delta_v_zeemans)))
    else:
        raise NotImplementedError
    
    #-- convolve with instrumental profile if desired
    if R is not None:
        instr_fwhm = wc/R
        instr_sigm = instr_fwhm/2.38
        logger.info('Convolving spectrum with instrumental profile of FWHM={:.3f}AA'.format(instr_fwhm))
        wavelengths,stokes_I = tools.rotational_broadening(wavelengths,
                       stokes_I/total_continum,vrot=0.,vmac=vmacro,fwhm=instr_sigm)
        stokes_I *= total_continum
        wavelengths,stokes_V = tools.rotational_broadening(wavelengths,
                       1-stokes_V/total_continum,vrot=0.,vmac=0.,fwhm=instr_sigm)
        stokes_V = (1-stokes_V)*total_continum
    
    if clip_after is not None:
        keep = (clip_after[0]<=wavelengths) & (wavelengths<=clip_after[1])
        wavelengths = wavelengths[keep]
        stokes_I = stokes_I[keep]
        stokes_V = stokes_V[keep]
        total_continum = total_continum[keep]
    return wavelengths,stokes_I,stokes_V,stokes_Q,stokes_U,total_continum
    

#}
#{ Input/output

def add_bitmap(system,image_file,select='teff',minval=None,maxval=None,res=1,
               update_intensities=False):
    """
    Add a pattern from a bitmap figure to a mesh column.
    
    At this point this function is for playing. Perhaps some day it can be
    useful.
    
    Use PNG! Not JPG!
    
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
    #-- normalise the coordinates so that we can map the coordinates
    PHI = (phi+np.pi)/(2*np.pi)*data.shape[0]
    THETA = (theta)/np.pi*data.shape[1]
    vals = np.array(ndimage.map_coordinates(data,[PHI,THETA]),float)
    #-- fix edges of image
    vals[PHI>0.99*PHI.max()] = 1.0
    vals[PHI<0.01*PHI.max()] = 1.0
    #-- rescale values between 0 and 1 so that we can map them between
    #   minval and maxval
    if minval is None: minval = system.mesh[select].min()
    if maxval is None: maxval = system.mesh[select].max()
    #-- don't map white values, but let the starlight shine through!
    keep = vals<0.99
    vals = (vals-vals.min())
    vals = vals/vals.max()
    vals = vals*(maxval-minval) + minval
    #-- and replace the values in the particular column
    system.mesh[select][keep] = vals[keep]
    if update_intensities:
        system.intensity()
    logger.info("Added bitmap {}".format(image_file))
            
def extract_times_and_refs(system,params,tol=1e-8):
    """
    Automatically extract times, references and types from a BodyBag.
    
    This function files the keys C{time}, C{types} and C{refs} in the
    parameterSet with context C{params}.
    
    If times are already set with a list, this function does nothing. This
    function only automatically fills the value if C{time='auto'}.\
    
    @param system: Body to derive time points of
    @type system: Body
    @param params: ParameterSet to fill in
    @type params: ParameterSet
    @param tol: tolerance limit for accepting two different time points as
    equal
    @type tol: float
    """
    #-- do we really need to set anything?
    dates = params['time']
    if not (isinstance(dates,str) and dates=='auto'):
        return None
    times = [] # time points
    types = [] # types
    refs  = [] # references
    #-- collect the times of the data, the types and refs of the data: walk
    #   through all the parameterSets, if they represent observations, load
    #   the dataset and collect the times. The reference and type for each
    #   parameterset is the same. Correct the time points to compute something
    #   on for exposure time and sampling rates: we add time points, which
    #   in the end should be then averaged over.
    for parset in system.walk():
        if not isinstance(parset,datasets.DataSet):
            continue
        if not parset.context[-3:]=='obs':
            continue
        loaded = parset.load(force=False)
        #-- exposure times
        if 'exptime' in parset:
            exps = parset['exptime']
        else:
            exps = [0]*len(parset['time'])
        #-- sampling rates
        if 'samprate' in parset:
            samp = parset['samprate']
        else:
            samp = [1]*len(parset['time'])
        for itime,iexp,isamp in zip(parset['time'],exps,samp):
            times.append(np.linspace(itime-iexp/2.0,itime+iexp/2.0,isamp))
            refs.append([parset['ref']]*isamp)
            types.append([parset.context]*isamp)
        if loaded: parset.unload()
    #-- what times are actually the same? We cannot test "true" equality
    #   because of possible numerical rounding (e.g. when reading or writing
    #   files). We test which times are "nearly" equal
    try:
        times = np.hstack(times)
    except ValueError as msg:
        raise ValueError("Failed to derive at which points the system needs to be computed. Perhaps the obs are not DataSets? (original message: {})".format(str(msg)))
    except IndexError as msg:
        raise ValueError("Failed to derive at which points the system needs to be computed. Perhaps there are no obs attached? (original message: {})".format(str(msg)))
    
    sa    = np.argsort(times)
    types = np.hstack(types)[sa]
    refs  = np.hstack(refs)[sa]
    times = times[sa]

    #-- for each time point, keep a list of stuff that needs to be computed
    labl_per_time = [] # observation ref (uuid..)
    type_per_time = [] # observation type (lcdep...)
    time_per_time = [] # times of observations
    for i,t in enumerate(times):
        #-- if the time is the first or different from the previous one:
        #   remember the time and start a list of refs and types
        if i==0 or np.abs(t-time_per_time[-1])>=tol:
            time_per_time.append(times[i])
            labl_per_time.append([refs[i]])
            type_per_time.append([types[i]])
        #-- else, append the refs and times to the last time point
        elif labl_per_time[-1][-1]!=refs[i]:
            labl_per_time[-1].append(refs[i])
            type_per_time[-1].append(types[i])
        else:
            #-- don't know what to do here: also append or continue?
            #continue
            labl_per_time[-1].append(refs[i])
            type_per_time[-1].append(types[i])
    #-- and fill the parameterSet!
    params['time'] = time_per_time
    params['refs']= labl_per_time
    params['types'] = type_per_time

@decorators.mpirun
def compute(system,params=None,**kwargs):
    """
    Automatically compute dependables of a system to match the observations.
    
    This is typically want you want to do if you have some data and want to
    compute a model generating those data. The times, references and types at
    which to compute anything, will be derived from the observations attached
    to the ``system``.
    
    Detailed configuration of the computations is provided via the optional
    parameterSet :ref:`compute <parlabel-phoebe-compute>`: ::
    
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

    But for convenience, all parameters in this parameterSet can also be
    given as kwargs.
        
    You can give an optional :ref:`mpi <parlabel-phoebe-mpi>` parameterSet.
    
    **Example usage**:
    
    First we quickly create a Star like Vega and add multicolour photometry:
    
    >>> vega = phoebe.create.from_library('vega')
    >>> mesh = phoebe.ParameterSet(context='mesh:marching')
    >>> lcdeps, lcobs = phoebe.parse_phot('vega.phot')
    >>> vega = phoebe.Star(vega,mesh,pbdep=lcdeps,obs=lcobs)
    
    Then we can easily compute the photometry to match the observations:
    
    >>> compute(vega)
    >>> compute(vega,subdiv_num=2)
    
    
    
    @param system: the system to compute
    @type system: Body
    @param params: computational parameterset
    @type params: ParameterSet of context ``compute``.
    @param mpi: parameters describing MPI
    @type mpi: ParameterSet of context 'mpi'
    """
    #-- gather the parameters that give us more details on how to compute
    #   the system: subdivisions, eclipse detection, optimization flags...
    #inside_mpi = kwargs.pop('mpi',None)
    im = kwargs.pop('im',False)
    extra_func = kwargs.pop('extra_func',[])
    extra_func_kwargs = kwargs.pop('extra_func_kwargs',[{}])
        
    if params is None:
        params = parameters.ParameterSet(context='compute',**kwargs)
    else:
        params = params.copy()
        for key in kwargs:
            params[key] = kwargs[key]
    auto_detect_circular = True
    #-- extract info on the time points to compute the system on, and which
    #   pbdeps (refs+types) to compute the system for. In principle, we
    #   could derive the type from the ref since they are unique, but this
    #   way we allow for a possibility to implement 'all lcdep' or so in the
    #   future.
    extract_times_and_refs(system,params)
    time_per_time = params['time']
    labl_per_time = params['refs']
    type_per_time = params['types'] 
    
    #-- some simplifications: try to detect whether a system is circular is not
    if hasattr(system,'bodies') and 'orbit' in system.bodies[0].params and auto_detect_circular:
        circular = (system.bodies[0].params['orbit']['ecc']==0)
        logger.info("Figured out that system is{0}circular".format(circular and ' ' or ' not '))
        #-- perhaps one of the components is a star with a spot or pulsations
        for body in system.bodies:
            if 'puls' in body.params or 'circ_spot' in body.params:
                circular = False
                logger.info("... but at least one of the components has spots or pulsations: set circular=False")
                break
            elif 'orbit' in body.params and 'theta' in body.params['orbit']: # it is misaligned
                circular = False
                logger.info("... but at least one of the components is misaligned")
                break
            
        
    else:
        circular = False
        logger.info("Cannot figure out if system is circular or not, leaving at circular={}".format(circular))
    #-- should we bother with irradiating Bodies and irradiated bodies? There
    #   are a few cases here: if the system is circular, we only need to
    #   compute the reflection once. In any case, we need to compute
    #   bolometric stuff at the desired time stamps (only once for circular
    #   cases, otherwise always)
    heating = params['heating'] 
    reflect = params['refl']
    nreflect = params['refl_num']
    ltt = params['ltt']
    
    #   so what about heating then...
    if heating and circular:
        heating = 1
        labl_per_time[0].append('__bol')
    elif heating:
        for labl in labl_per_time:
            labl.append('__bol')
    #   and uuhhh... what about reflection?
    if reflect and circular:
        reflect = 1
        if not heating:
            labl_per_time[0].append('__bol')
    elif reflect and not heating:
        for labl in labl_per_time:
            labl.append('__bol')
    #   and don't forget beaming!
    beaming = False
    for parset in system.walk():
        if 'beaming' in parset and parset['beaming']:
            beaming = True
            logger.info("Figured out that the system requires beaming")
    #-- if we include reflection, we need to reserve space in the mesh
    #   for the reflected light. We need to fix the mesh afterwards because
    #   each body can have different fields appended in the mesh.
    if reflect:
        system.prepare_reflection(ref='all')
        x1 = set(system[0].mesh.dtype.names)
        x2 = set(system[1].mesh.dtype.names)
        if len(x1-x2) or len(x2-x1):
            #raise ValueError("When including reflection, you need to call 'prepare_reflection' and 'fix_mesh' first")
            system.fix_mesh()
    #-- now we're ready to do the real stuff
    for i,(time,ref,type) in enumerate(zip(time_per_time,labl_per_time,type_per_time)):
        #-- clear previous reflection effects if necessary (not if reflect==1!)
        if reflect is True:
            system.clear_reflection()
        #-- set the time of the system
        system.set_time(time,ref=ref)
        #-- fix the mesh if needed:
        if i==0 and hasattr(system,'bodies'):
            system.fix_mesh()
        #-- for heating an eccentric system, we first need to reset the temperature!
        if heating is True:
            system.temperature(time)
        #-- compute intensities
        if i==0 or not circular or beaming:
            system.intensity(ref=ref)
        #-- update intensity should be set to True when we're doing beaming.
        #   Perhaps we need to detect which refs have "beaming=True", collect
        #   those in a list and update the intensities for them anyway?
        update_intensity = False
        #-- compute reflection effect (maybe just once, maybe always)
        if (reflect is True or heating is True) or (i==0 and (reflect==1 or heating==1)):
            reflection.mutual_heating(*system.bodies,heating=heating,
                                      reflection=reflect,niter=nreflect)
            update_intensity = True
        #-- recompute the intensities (the velocities might have
        #   changed within BodyBag operations, and temperatures might
        #   have changed due to reflection)
        if update_intensity:
            system.intensity(ref=ref)
        #-- detect eclipses/horizon
        choose_eclipse_algorithm(system,algorithm=params['eclipse_alg'])
        #-- if necessary, subdivide and redetect eclipses/horizon
        for k in range(params['subdiv_num']):
            system.subdivide(threshold=0,algorithm=params['subdiv_alg'])
            choose_eclipse_algorithm(system,algorithm=params['eclipse_alg'])
        #-- correct for light travel time effects
        if ltt:
            system.correct_time()
        #-- compute stuff
        had_refs = [] # we need this for the ifm, so that we don't compute stuff too much
        for itype,iref in zip(type,ref):
            if itype[:-3]=='if':
                itype = 'ifmobs' # would be obsolete if we just don't call it "if"!!!
                if iref in had_refs: continue
                had_refs.append(iref)
            logger.info('Calling {} for ref {}'.format(itype[:-3],iref))
            getattr(system,itype[:-3])(ref=iref,time=time)
        #-- make an image if necessary
        if im:
            if isinstance(im,str):
                savefig = ('%s_%014.6f.png'%(im,time)).replace('.','_')
            else:
                savefig = ('compute_dependable_%014.6f.png'%(time)).replace('.','_')
            image(system,ref=ref[0],savefig=savefig)
        #-- call extra funcs if necessary
        for ef,kw in zip(extra_func,extra_func_kwargs):
            ef(system,time,i,**kw)
        #-- unsubdivide to prepare for next step
        if params['subdiv_num']:  
            system.unsubdivide()
    #if inside_mpi is None:
    try:
        system.compute_pblum_or_l3()
    except:
        logger.warning("Cannot compute pblum or l3. I can think of two reasons why this would fail: (1) you're in MPI (2) you have previous results attached to the body.")


def observe(system,times,lc=False,rv=False,sp=False,pl=False,im=False,mpi=None,
            extra_func=[],extra_func_kwargs=[{}],**kwargs):
    """
    Customized computation of dependables of a system.
    
    This is similar as :py:func:`compute`. The difference
    is that this function is more flexible. You have to provide your own times,
    but also what type of observations you want to compute. This is probably
    required when you want to do simulations of a system, without comparing
    the observations.
    
    Parameters to tweak the calculations are given as keyword arguments, and
    can be any of::
    
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
    """
    #-- gather the parameters that give us more details on how to compute
    #   the system: subdivisions, eclipse detection, optimization flags...
    params = parameters.ParameterSet(context='compute',**kwargs)
    #-- what do we need to compute? The user only has the possibility of
    #   saying e.g. lc=True or lc=['mylc1','mylc2']
    refs = []
    typs = []
    if not lc and not rv and not sp and not pl and not im:
        raise ValueError("You need to compute at least one of lc, rv, sp, pl and/or im")
    #-- derive all lcdep parameterset references
    for type in ['lc','rv','sp','pl']:
        if locals()[type] is True:
            for parset in system.walk():
                if parset.context==type+'dep':
                    if parset['ref'] in refs:
                        continue
                    refs.append(parset['ref'])
                    typs.append(type+'dep')
    #-- fill in the parameterSet
    params['time'] = times
    params['refs'] = [refs]*len(times)
    params['types'] = [typs]*len(times)  
    #-- and run compute
    compute(system,params=params,mpi=mpi,im=im,extra_func=extra_func,
            extra_func_kwargs=extra_func_kwargs)


        
def choose_eclipse_algorithm(all_systems,algorithm='auto'):
    """
    Try to automatically detect the best eclipse detection algorithm.
    """
    #-- perhaps we know there are no eclipses
    if algorithm=='only_horizon':
        eclipse.horizon_via_normal(all_systems)
        return None
    elif algorithm=='full':
        eclipse.detect_eclipse_horizon(all_systems)
        
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
                if algorithm=='auto':
                    eclipse.detect_eclipse_horizon(all_systems)
                elif algorithm=='convex':
                    eclipse.convex_bodies([all_systems[0],all_systems[1]])
            else:
                logger.info("{}: predict no eclipse (simple E/H)".format(all_systems[0].time))
                eclipse.horizon_via_normal(all_systems)
        elif hasattr(all_systems,'len') and len(all_systems)>1: # assume it's a multiple system
            logger.info("{}: predict eclipse (generic E/H)".format(all_systems[0].time))
            if algorithm=='auto':
                eclipse.detect_eclipse_horizon(all_systems)
            elif algorithm=='convex':
                eclipse.convex_bodies(all_systems.get_bodies())
            #eclipse.detect_eclipse_horizon(all_systems)
        elif hasattr(all_systems,'params') and 'component' in all_systems.params:
            logger.warning('Perhaps (over)contact system (generic E/H)')
            eclipse.detect_eclipse_horizon(all_systems)
        else: # single star, easy detection!
            logger.info("simple E/H detection")
            eclipse.horizon_via_normal(all_systems)
    except ValueError:
        logger.info('Could not interpret system, (generic E/H)')
        if algorithm=='auto':
            eclipse.detect_eclipse_horizon(all_systems)
        elif algorithm=='convex':
            eclipse.convex_bodies(all_systems.get_bodies())

#{ Extrafuncs for compute_dependables

def ef_binary_image(system,time,i,name='ef_binary_image',**kwargs):
    """
    Make an image of a binary system.
    
    But setting the x and y limits to sensible values, so that
    we can always see the entire orbit. This eliminates the zoom effect in the
    default behaviour, but of course we need to know the maximum size of the system
    without computing through it. For binary systems, this is of course fairly
    easy.
    """
    # Compute the orbit of the system
    if hasattr(system,'__len__'):
        orbit = system[0].params['orbit']
        star1 = system[0]
        star2 = system[1]
    else:
        orbit = system.params['orbit']
        star1 = system
        star2 = None
    period = orbit['period']
    t0,tn = kwargs.pop('t0',0), kwargs.pop('tn',period)
    times_ = np.linspace(t0,tn,250)
    orbit1 = keplerorbit.get_binary_orbit(times_,orbit,component='primary')[0]
    orbit2 = keplerorbit.get_binary_orbit(times_,orbit,component='secondary')[0]
    # What's the radius of the stars?
    r1 = coordinates.norm(star1.mesh['_o_center'],axis=1).mean()
    if star2 is not None:
        r2 = coordinates.norm(star2.mesh['_o_center'],axis=1).mean()
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
    #pl.plot(orbit1[0],orbit1[1],'r-',lw=2)
    #pl.plot(orbit2[0],orbit2[1],'b-',lw=2)
    pl.xlim(xmin,xmax)
    pl.ylim(ymin,ymax)
    pl.savefig('{}_{:04d}'.format(name,i))
    pl.close()


def ef_image(system,time,i,name='ef_image',comp=0,**kwargs):
    """
    Make an image of a binary system.
    
    But setting the x and y limits to sensible values, so that
    we can always see the entire orbit. This eliminates the zoom effect in the
    default behaviour, but of course we need to know the maximum size of the system
    without computing through it. For binary systems, this is of course fairly
    easy.
    """
    # Get the thing to plot
    if hasattr(system,'__len__'):
        system = system[comp]
    # and make the figure
    kwargs['savefig'] = '{}_comp_{:02d}_{:04d}'.format(name,comp,i)
    image(system,**kwargs)



def plot_system(system):
    """
    Plot models and observations.        
    """
    for idata in system.params['obs'].values():
        for observations in idata.values():
            pl.figure()
            pl.title(observations['ref'])
            #-- get the model corresponding to this observation
            model = system.get_synthetic(type=observations.context[:-3]+'syn',
                                       ref=observations['ref'],
                                       cumulative=True)
            observations.plot()
            model.plot()




#}








        

    