"""
Non-standard interpolation methods.
"""
import time
import itertools
import libphoebe   
import numpy as np
from scipy import ndimage
from scipy.ndimage import _nd_image, _ni_support
try:
    import pyfits as pf
except:
    try: # Pyfits now integrated in astropy
        import astropy.io.fits as pf
    except ImportError:
        print(("Soft warning: pyfits could not be found on your system, you can "
           "only use black body atmospheres and Gaussian synthetic spectrallll "
           "profiles"))


def create_pixeltypegrid(grid_pars, grid_data):
    """
    Creates pixelgrid and arrays of axis values.
    
    Starting from:
    
        - grid_pars: 2D numpy array, 1 column per parameter, unlimited number of
          cols
        - grid_data: 2D numpy array, 1 column per variable, data corresponding
          to the rows in grid_pars
    
    
    The grid should be rectangular and complete, i.e. every combination of the unique values in the 
    parameter columns should exist. If not, a nan value will be inserted.
    
    @param grid_pars: Npar x Ngrid array of parameters
    @type grid_pars: array
    @param grid_data: Ndata x Ngrid array of data
    @type grid_data: array
    @return: axis values and pixelgrid
    @rtype: array, array
    """

    uniques = [np.unique(column, return_inverse=True) for column in grid_pars]
    #[0] are the unique values, [1] the indices for these to recreate the original array
    
    # we need to copy the values of the unique axes explicitly into new arrays
    # otherwise we can get issues with the interpolator
    axis_values = []
    for uniques_ in uniques:
        this_axes = np.zeros(len(uniques_[0]))
        this_axes[:] = uniques_[0]
        axis_values.append(this_axes)
    #axis_values = [uniques_[0] for uniques_ in uniques]
    #axis_values = [np.require(uniques_[0],requirements=['A','O','W','F']) for uniques_ in uniques]
    
    unique_val_indices = [uniques_[1] for uniques_ in uniques]
    
    data_dim = np.shape(grid_data)[0]

    par_dims   = [len(uv[0]) for uv in uniques]

    par_dims.append(data_dim)
    pixelgrid = np.ones(par_dims)
    
    # We put np.inf as default value. If we get an inf, that means we tried to access
    # a region of the pixelgrid that is not populated by the data table
    pixelgrid[pixelgrid==1] = np.inf
    
    # now populate the multiDgrid
    indices = [uv[1] for uv in uniques]
    pixelgrid[indices] = grid_data.T
    return tuple(axis_values), pixelgrid


def interpolate(p, axis_values, pixelgrid, order=1, mode='constant', cval=0.0):
    """
    Interpolates in a grid prepared by create_pixeltypegrid().
    
    p is an array of parameter arrays
    
    @param p: Npar x Ninterpolate array
    @type p: array
    @return: Ndata x Ninterpolate array
    @rtype: array
    """
    # convert requested parameter combination into a coordinate
    p_ = np.array([np.searchsorted(av_,val) for av_, val in zip(axis_values,p)])
    lowervals_stepsize = np.array([[av_[p__-1], av_[p__]-av_[p__-1]] \
                         for av_, p__ in zip(axis_values,p_)])
    p_coord = (p-lowervals_stepsize[:,0])/lowervals_stepsize[:,1] + p_-1
    
    # interpolate
    if order > 1:
        prefilter = False
    else:
        prefilter = False

    return [ndimage.map_coordinates(pixelgrid[...,i],p_coord, order=order,
                                    prefilter=prefilter, mode=mode, cval=cval) \
                for i in range(np.shape(pixelgrid)[-1])]


def cinterpolate(p, axis_values, pixelgrid):
    """
    Interpolates in a grid prepared by create_pixeltypegrid().
    
    Does a similar thing as :py:func:`interpolate`, but does everything in C.
    
    p is an array of parameter arrays.
    
    Careful, the shape of input :envvar:`p` and output is the transpose of
    :py:func:`interpolate`.
    
    @param p: Ninterpolate X Npar array
    @type p: array
    @return: Ninterpolate X Ndata array
    @rtype: array
    """
    res = libphoebe.interp(p, axis_values, pixelgrid)
    return res


if __name__ == "__main__":
    
    # mock data
    x = np.linspace(0,10,20)
    y = np.linspace(0,10,30)
    z = np.linspace(0,10,30)
    a = np.linspace(0,10,30)
    grid_pars = []
    
    for pars in itertools.product(x,y,z,a):
        grid_pars.append(pars)    
    grid_pars = np.array(grid_pars).T
    grid_data = np.array([1000000*grid_pars[0]+10000*grid_pars[1]+100*grid_pars[2]+grid_pars[3],100000*grid_pars[0]+1000*grid_pars[1]+10*grid_pars[2]+grid_pars[3]/10.,1000000*grid_pars[0]+10000*grid_pars[1]+100*grid_pars[2]+grid_pars[3],1000000*grid_pars[0]+10000*grid_pars[1]+100*grid_pars[2]+grid_pars[3]])
    print(np.shape(grid_data))
    p = np.array([[1]*10000,[1.2]*10000,[3.1]*10000,[0.1]*10000])
    print("Creating array")
    c0=time.time()    
    axis_values, pixelgrid = create_pixeltypegrid(grid_pars, grid_data)
    c1= time.time()
    print("Interpolating")
    vals = interpolate(p, axis_values,pixelgrid)
    
    c2=time.time()
    
    print("Creating array:", c1-c0, 's')
    print("Interpolation:", c2-c1, 's')
    print(vals)
    # ---------------------------------------------------------------------
