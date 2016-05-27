
import numpy as np
from phoebe.algorithms import ceclipse
import logging

logger = logging.getLogger("ECLIPSE")

def _graham_scan_inside_hull(front, back):
    sa = np.argsort(front[:,0], kind='heapsort')
    hull, inside = ceclipse.graham_scan_inside_hull(front[sa], back)
    return hull, inside

"""
each of these functions needs to take meshes, xs, ys, zs.
- meshes is a dictionary with keys being the component number of that mesh and
  values being the meshes themselves
- xs, ys, zs are lists of the positions of each component (ie xs[comp_no-1])
each function must return a dictionary, with keys being component numbers and values
  being the revised visibilities of the corresponding mesh

"""

def wd_horizon(meshes, xs, ys, zs):
    """
    """

    nbodies = len(meshes.keys())

    if nbodies != 2:
        raise ValueError("wd_horizon eclipse method only works on binaries")

    front_to_back_comp_nos = np.argsort(zs)[::-1]+1

    comp_front = meshes.component_by_no(front_to_back_comp_nos[0])
    comp_back = meshes.component_by_no(front_to_back_comp_nos[1])

    mesh_front = meshes[comp_front]
    mesh_back = meshes[comp_back]

    rhos = [np.sqrt((mesh['center'][:,0]-xs[i])**2 + (mesh['center'][:,1]-ys[i])**2) for i,mesh in enumerate([mesh_front, mesh_back])]
    thetas = [np.arcsin((mesh['center'][:,1]-ys[i])/rhos[i]) for i,mesh in enumerate([mesh_front, mesh_back])]

    import matplotlib.pyplot as plt
    plt.plot(rhos[0], thetas[0], 'b.')
    plt.plot(rhos[1], thetas[1], 'r.')
    plt.show()


    visibilities = {comp_no: np.ones(len(mesh)) for comp_no, mesh in meshes.items()}

    return visibilities


def none(meshes, xs, ys, zs):
    """
    """
    return {comp_no: mesh['visibility'] for comp_no, mesh in meshes.items()}

def only_horizon(meshes, xs, ys, zs):
    """
    Check all visible or partial triangles to see if they're behind the horizon,
    by checking the direction of the z-component of the normals (ie hidden if mu<0)
    """

    # if visibility == 0, it should remain 0
    # if visibility == 0.5, it should stay 0.5 if mu > 0 else it should become 0
    # if visibility == 1, it should stay 1 if mu > 0 else it should become 0

    # this can all by easily done by multiplying by int(mu>0) (1 if visible, 0 if hidden)

    return {comp_no: mesh['visibility'] * (mesh['mu'] > 0).astype(int) for comp_no, mesh in meshes.items()}


def graham(meshes, xs, ys, zs):
    """
    convex_graham
    """
    distance_factor = 1.0  # TODO: make this an option (what does it even do)?

    # first lets handle the horizon
    visibilities = only_horizon(meshes, xs, ys, zs)

    # Order the bodies from front to back.  We can do this through zs (which are
    # the z-coordinate positions of each body in the system.
    # Those indices are in the same order as the meshes['comp_no']
    # (needing to add 1 because comp_no starts at 1 not 0)

    # TODO: I don't think this whole comp_no thing is going to work with nested
    # triples.. may need to rethink the whole meshes.component_by_no thing

    front_to_back_comp_nos = np.argsort(zs)[::-1]+1
    nbodies = len(front_to_back_comp_nos)

    for i_front in range(0, nbodies-1):
        # for a binary, i_front will only be 0
        comp_no_front = front_to_back_comp_nos[i_front]
        comp_front = meshes.component_by_no(comp_no_front)
        mesh_front = meshes[comp_front]
        visibility_front = visibilities[comp_front]

        for i_back in range(i_front+1, nbodies):
            # for a binary, i_back will only be 1
            comp_no_back = front_to_back_comp_nos[i_back]
            comp_back = meshes.component_by_no(comp_no_back)
            mesh_back = meshes[comp_back]
            visibility_back = visibilities[comp_back]

            # If mesh_back is entirely hidden, then we can skip any checks and
            # leave it hidden.  Note that here we need to check visibility instead
            # of mesh['visibility'] or mesh_back['visibility'] since we are not
            # adjusting those values in memory.
            if np.all(visibility_back==0.0):
                continue

            # Determine a scale factor for the triangle
            min_size_back = mesh_back['size'].min()
            distance = distance_factor * 2.0/3**0.25*np.sqrt(min_size_back)

            # Select only those triangles that are not hidden
            tri_back_vis = mesh_back['triangle'][visibility_back > 0.0]
            tri_front_vis = mesh_front['triangle'][visibility_front > 0.0]

            back = np.vstack([tri_back_vis[:,0:2], tri_back_vis[:,3:5], tri_back_vis[:,6:8]])
            front = np.vstack([tri_front_vis[:,0:2], tri_front_vis[:,3:5], tri_front_vis[:,6:8]])

            # Star in front ---> star in back
            if not front.shape[0]:
                continue

            hull, inside = _graham_scan_inside_hull(front, back)
            hidden = inside.reshape(3,-1).all(axis=0)
            visible = ~(inside.reshape(3,-1).any(axis=0))

            # Triangles that are partially hidden are those that are not
            # completely hidden, but do have at least one vertex hidden
            partial = ~hidden & ~visible

            # These returned visibilities are only from mesh_back_vis
            # So to apply back to our master visibilities parameter, we need
            # to find the correct inds.
            visibility_back[visibility_back > 0.0] = 1.0*visible + 0.5*partial


            ###################################################
            ## TODO: port the following code to be included  ##
            ###################################################


            # It's possible that some triangles that do not have overlapping
            # vertices, still are partially covered (i.e. if the objects fall
            # in between the vertices)
            #visible1 = ~star1.mesh['hidden']
            #backc = star1.mesh['center'][visible1,0:2]
            #if not len(backc):
            #    continue
            #
            #subset_partial = closest_points(hull, backc, distance)
            #
            #if not len(subset_partial):
            #    continue
            #
            #arg = np.where(visible1)[0][subset_partial]
            #star1.mesh['visible'][arg] = False
            #star1.mesh['partial'][arg] = True


    return visibilities
