"""
Algorithms to subdivide meshes.
"""
import logging
import numpy as np
from phoebe.algorithms import fsubdivision

logger = logging.getLogger("ALG.SUBDIV")

def subdivide(old_mesh,prefix='',threshold=0,algorithm='edge'):
    """
    Subdivision of a mesh via triangle edges or triangle's centers.
    
    Subidividing via triangle edges (C{algorithm='edge'}) guarantees well-formed
    triangles, but does not allow for a reprojection of the surface, since holes
    can occur. Subdividing via the triangle's centers (C{algorithm='center'})
    allows reprojection of the surface, but does not guarentee well-formed
    triangles.
    
    Only returns the partial triangles!
    
    Prefix discriminates between subdividing triangles in original reference
    frame ('_o_') or subdividing triangles in the current reference frame ('').
    If you want to do both, you can give a list, and I guess it's better to
    first give the '_o_' and then the '' version.
    
    """
    if algorithm=='edge':
        Nincrease = 4
    elif algorithm=='center':
        Nincrease = 3
    N = Nincrease*len(old_mesh)
    if N==0:
        return np.zeros(0,dtype=old_mesh.dtype)
    #if debug: print "... constructing %d new triangles"%(N)
    new_triangles = np.zeros((N,old_mesh['triangle'].shape[1]))
    new_centers   = np.zeros((N,old_mesh['center'].shape[1]))
    new_sizes     = np.zeros(N)
    new_normals   = np.zeros((N,old_mesh['normal_'].shape[1]))
    new_mu        = np.zeros(N)
    indices = np.hstack(Nincrease*[np.arange(0,len(old_mesh),1)]).ravel()
    new_mesh      = old_mesh[indices]
    #-- subdivide: we might want to do it both for the original and for the
    #   "current" mesh.
    if isinstance(prefix,str):
        prefix = [prefix]
    for iprefix in prefix:
        if algorithm=='edge':
            new_triangles,new_sizes,new_normals,new_centers,new_mu = \
                fsubdivision.simple_subdivide(old_mesh[iprefix+'triangle'],old_mesh[iprefix+'size'],
                                old_mesh[iprefix+'normal_'],old_mesh['mu'],threshold,new_triangles,
                                new_sizes,new_normals,new_centers,new_mu)
        elif algorithm=='center':
            new_triangles,new_sizes,new_normals,new_centers,new_mu = \
                fsubdivision.simple_subdivide_middle(old_mesh[iprefix+'triangle'],old_mesh[iprefix+'size'],
                                old_mesh[iprefix+'normal_'],old_mesh['mu'],threshold,new_triangles,
                                new_sizes,new_normals,new_centers,new_mu)
        else:
            raise ValueError('do not understand algorithm=%s'%(algorithm))
        #-- replace old triangles with newly subdivided ones
        new_mesh['mu'] = new_mu
        new_mesh[iprefix+'center'] = new_centers
        new_mesh[iprefix+'triangle'] = new_triangles
        new_mesh[iprefix+'size'] = new_sizes
        new_mesh[iprefix+'normal_'] = new_normals
        new_mesh['partial'] = True
        
    #-- only keep the ones with nonzero sizes (perhaps not all triangles were
    #   subdivided):
    keep = (new_sizes!=0) & -np.isnan(new_sizes)
    new_mesh = new_mesh[keep]
    logger.debug("subdivided via %s (skipped %d of %d new triangles because they were too small)"%(algorithm,N-len(new_mesh),N))
    return new_mesh