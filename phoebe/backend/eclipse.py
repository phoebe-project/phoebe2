import sys
import numpy as np

from phoebe.algorithms import ceclipse
from phoebe.utils import _bytes
import libphoebe

import logging

_EXPORT_HORIZON = False

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

def wd_horizon(meshes, xs, ys, zs, expose_horizon=False):
    """
    """

    nbodies = len(meshes.keys())

    if nbodies != 2:
        raise ValueError("wd_horizon eclipse method only works on binaries")

    front_to_back_comp_nos = np.argsort(zs)[::-1]
    i_front = front_to_back_comp_nos[0]
    i_back = front_to_back_comp_nos[1]

    comp_front = meshes.component_by_no(i_front+1)
    comp_back = meshes.component_by_no(i_back+1)

    mesh_front = meshes[comp_front]
    mesh_back = meshes[comp_back]


    # polar coordinates need to be wrt the center of the ECLIPSING body
    rhos = [np.sqrt((mesh.centers[:,0]-xs[i_front])**2 +
                    (mesh.centers[:,1]-ys[i_front])**2) for mesh in (mesh_front, mesh_back)]
    # thetas = [np.arcsin((mesh['center'][:,1]-ys[i_front])/rho) for mesh,rho in zip((mesh_front, mesh_back), rhos)]
    thetas = [np.arctan2(mesh.centers[:,1]-ys[i_front], mesh.centers[:,0]-xs[i_front]) for mesh in (mesh_front, mesh_back)]
    # mus = [mesh['mu'] for mesh in (mesh_front, mesh_back)]

    # import matplotlib.pyplot as plt
    # plt.plot(rhos[0], thetas[0], 'b.')
    # plt.plot(rhos[1], thetas[1], 'r.')
    # plt.show()

    # To find the horizon we want the first positive mu elements (on right and
    # left) for each latitude strip.

    horizon_inds = []
    # we only need the horizon of the ECLIPSING star, so we'll use mesh_front
    # and mus[i_front], xs[i_front], etc
    lats = list(set(mesh_front.theta))
    for lat in lats:
        lat_strip_inds = mesh_front.theta == lat

        # let's get the x-coordinate wrt THIS star so we can do left vs right
        x_rel = mesh_front.centers[:,0] - xs[i_front]

        print(lat, x_rel[lat_strip_inds].min(), x_rel[lat_strip_inds].max())

        # and since we want the first element in the front, let's just get rid of the back
        front_inds = mesh_front.mus >= 0.0
        back_inds = mesh_front.mus < 0.0

        left_inds = x_rel < 0.0
        right_inds = x_rel >= 0.0

        # let's handle "left" vs "right" side of star separately
        for side_inds, fb_inds, side in zip((left_inds, right_inds), (front_inds, back_inds), ('left', 'right')):
            no_triangles = len(mesh_front.mus[lat_strip_inds * side_inds * fb_inds])
            # print "*** no triangles at lat", lat, no_triangles

            if no_triangles > 0:
                if side=='left':
                    # then we want the first triangle on the FRONT of the star
                    first_horizon_mu = mesh_front.mus[lat_strip_inds * side_inds * fb_inds].min()
                else:
                    # then we want the first triangle on the BACK of the star
                    first_horizon_mu = mesh_front.mus[lat_strip_inds * side_inds * fb_inds].max()

                first_horizon_ind = np.where(mesh_front.mus==first_horizon_mu)[0][0]
                # print "*** horizon index", first_horizon_ind

                horizon_inds.append(first_horizon_ind)

    thetas[0][thetas[0] < 0] = thetas[0][thetas[0] < 0]+2*np.pi

    f = open('bla2', 'w')
    for ind in horizon_inds:
        # note here rhos and thetas need 0 not i_front because of the stupid way we did the list comprehension
        f.write("{} {} {}\n".format(ind, rhos[0][ind], thetas[0][ind]))
    f.close()

    # these are the horizon coordinates for the ECLIPSING star
    rhos[0][horizon_inds]
    thetas[0][horizon_inds]

    visibilities, weights, horizon = only_horizon(meshes, xs, ys, zs)
    # now edit visibilities based on eclipsing region
    # visibilities = {comp_no: np.ones(len(mesh)) for comp_no, mesh in meshes.items()}
    visibilities[comp_back]
    mesh_back

    return visibilities, None, None


def none(meshes, xs, ys, zs, expose_horizon=False):
    """
    """
    return {comp_no: mesh.visibilities for comp_no, mesh in meshes.items()}, None, None

def only_horizon(meshes, xs, ys, zs, expose_horizon=False):
    """
    Check all visible or partial triangles to see if they're behind the horizon,
    by checking the direction of the z-component of the normals (ie hidden if mu<0)
    """

    # if visibility == 0, it should remain 0
    # if visibility == 0.5, it should stay 0.5 if mu > 0 else it should become 0
    # if visibility == 1, it should stay 1 if mu > 0 else it should become 0

    # this can all by easily done by multiplying by int(mu>0) (1 if visible, 0 if hidden)

    return {comp_no: mesh.visibilities * (mesh.mus > 0).astype(int) for comp_no, mesh in meshes.items() if mesh is not None}, None, None

def native(meshes, xs, ys, zs, expose_horizon=False, horizon_method='boolean'):
    """
    TODO: add documentation

    this is the new eclipse detection method in libphoebe
    """

    centers_flat = meshes.get_column_flat('centers')
    vertices_flat = meshes.get_column_flat('vertices')
    triangles_flat = meshes.get_column_flat('triangles')  # should handle offset automatically

    if horizon_method=='boolean':
        normals_flat = meshes.get_column_flat('tnormals')
    elif horizon_method=='linear':
        normals_flat = meshes.get_column_flat('vnormals')
    else:
        raise NotImplementedError

    # viewing_vector is defined as star -> earth
    # NOTE: this will need to flip if we change the convention on the z-direction
    viewing_vector = np.array([0., 0., 1.])


    # we need to send in ALL vertices but only the visible triangle information
    info = libphoebe.mesh_visibility(viewing_vector,
                                     vertices_flat,
                                     triangles_flat,
                                     normals_flat,
                                     tvisibilities=True,
                                     taweights=True,
                                     method=_bytes(horizon_method),
                                     horizon=expose_horizon)

    visibilities = meshes.unpack_column_flat(info['tvisibilities'], computed_type='triangles')
    weights = meshes.unpack_column_flat(info['taweights'], computed_type='triangles')

    if expose_horizon:
        horizons = info['horizon']

        # TODO: we need to do this per component and somehow return them in
        # a predictable order or as a dictionary a la the other returned quantities
        horizons = [vertices_flat[horizon_i] for horizon_i in horizons]

    else:
        horizons = None

    return visibilities, weights, horizons

def visible_partial(meshes, xs, ys, zs, expose_horizon=False):
    centers_flat = meshes.get_column_flat('centers')
    vertices_flat = meshes.get_column_flat('vertices')
    triangles_flat = meshes.get_column_flat('triangles')  # should handle offset automatically
    normals_flat = meshes.get_column_flat('tnormals')

    # viewing_vector is defined as star -> earth
    # NOTE: this will need to flip if we change the convention on the z-direction
    viewing_vector = np.array([0., 0., 1.])

    visibilities = libphoebe.mesh_rough_visibility(viewing_vector,
                                                   vertices_flat,
                                                   triangles_flat,
                                                   normals_flat)

    visibilities = meshes.unpack_column_flat(visibilities, computed_type='triangles')

    return visibilities, None, None



def graham(meshes, xs, ys, zs, expose_horizon=False):
    """
    convex_graham
    """

    distance_factor = 1.0  # TODO: make this an option (what does it even do)?

    # first lets handle the horizon
    visibilities, weights, horizon = only_horizon(meshes, xs, ys, zs)

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
            min_size_back = mesh_back.areas.min()
            distance = distance_factor * 2.0/3**0.25*np.sqrt(min_size_back)

            # Select only those triangles that are not hidden
            tri_back_vis = mesh_back.vertices_per_triangle[visibility_back > 0.0].reshape(-1,9)
            tri_front_vis = mesh_front.vertices_per_triangle[visibility_front > 0.0].reshape(-1,9)

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


    return visibilities, None, None
