"""
Algorithms to subdivide meshes.
"""
import logging
import numpy as np
from phoebe.algorithms import fsubdivision

logger = logging.getLogger("SUBDIVISION")

def _create_new_arrays(old_mesh, Nincrease):
    """
    """

    N = Nincrease * len(old_mesh)

    new_triangles = np.zeros((N,old_mesh['triangle'].shape[1]))
    new_centers   = np.zeros((N,old_mesh['center'].shape[1]))
    new_sizes     = np.zeros(N)
    new_normals   = np.zeros((N,old_mesh['normal_'].shape[1]))
    new_mu        = np.zeros(N)

    return new_triangles, new_centers, new_sizes, new_normals, new_mu

def _increase_submesh_size(old_mesh, Nincrease):
    """
    """

    N = Nincrease * len(old_mesh)

    indices = np.hstack(Nincrease*[np.arange(0,len(old_mesh),1)]).ravel()
    
    # new_mesh will be a copy of old_mesh but with each element duplicated
    # N times.  Any column which we don't overwrite will remain as a copy
    # of its parent triangle.
    new_mesh = old_mesh[indices]

    return new_mesh

def _apply_new_arrays(new_mesh, new_triangles, new_centers, new_sizes, new_normals, new_mu):
    """
    """
    new_mesh['mu'] = new_mu
    new_mesh['center'] = new_centers
    new_mesh['triangle'] = new_triangles
    new_mesh['size'] = new_sizes
    new_mesh['normal_'] = new_normals

    return new_mesh

def _remove_zeros(new_mesh, new_sizes):
    
    keep = (new_sizes!=0) & ~np.isnan(new_sizes)
    new_mesh = new_mesh[keep]
    #logger.debug("subdivided (skipped %d of %d new triangles because they were too small)"% (N-len(new_mesh),N))
    return new_mesh


def edge(mesh, threshold=0):
    """
    """
    Nincrease = 4

    # First we need to initialize the arrays so we can send them to the fortran
    # routine
    new_triangles, new_centers, new_sizes, new_normals, new_mu = \
        _create_new_arrays(mesh, Nincrease)

    # Now we can pass those empty np.zeros arrays to the fortran routine
    new_triangles, new_sizes, new_normals, new_centers, new_mu = \
        fsubdivision.simple_subdivide(mesh['triangle'], mesh['size'],
                        mesh['normal_'], mesh['mu'], threshold, new_triangles,
                        new_sizes, new_normals, new_centers, new_mu)

    # Now let's expand the original mesh so that each element is copied as necessary
    new_mesh = _increase_submesh_size(mesh, Nincrease)

    # And lastly, we need to update the geometric quantities to be of the subdivided
    # triangles.  All columns we don't touch will remain exact copies of the parent
    # triangle.
    new_mesh = _apply_new_arrays(new_mesh, new_triangles, new_centers, new_sizes, new_normals, new_mu)

    new_mesh = _remove_zeros(new_mesh, new_sizes)

    return new_mesh


def center(mesh, threshold=0):
    """
    """
    Nincrease = 3

    # First we need to initialize the arrays so we can send them to the fortran
    # routine
    new_triangles, new_centers, new_sizes, new_normals, new_mu = \
        _create_new_arrays(mesh, Nincrease)

    # Now we can pass those empty np.zeros arrays to the fortran routine
    new_triangles, new_sizes, new_normals, new_centers, new_mu = \
            fsubdivision.simple_subdivide_middle(mesh['triangle'], mesh['size'],
                            mesh['normal_'], mesh['mu'], threshold, new_triangles,
                            new_sizes, new_normals, new_centers, new_mu)

    # Now let's expand the original mesh so that each element is copied as necessary
    new_mesh = _increase_submesh_size(mesh, Nincrease)

    # And lastly, we need to update the geometric quantities to be of the subdivided
    # triangles.  All columns we don't touch will remain exact copies of the parent
    # triangle.
    new_mesh = _apply_new_arrays(new_mesh, new_triangles, new_centers, new_sizes, new_normals, new_mu)

    new_mesh = _remove_zeros(new_mesh, new_sizes)

    return new_mesh
