import numpy as np
from scipy.optimize import curve_fit as cfit

from phoebe.algorithms import ceclipse
import libphoebe

import logging

_EXPORT_HORIZON = True

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

    def fourier(theta, *c):
        N = len(c)/2

        rho = 0
        for k in range(N):
            rho += c[k]*np.cos(k*theta) + c[N+k]*np.sin(k*theta)

        return rho


    def correction(rhos_back, thetas_back, ind, *coeffs):
        dr1 = rhos_back[ind]-fourier(thetas_back[ind], *coeffs)
        dr2 = rhos_back[ind-1]-fourier(thetas_back[ind-1], *coeffs)
        rind = ind-1
        if dr1*dr2 > 0 and ind+1 < len(rhos_back):
            dr2 = rhos_back[ind+1]-fourier(thetas_back[ind+1], *coeffs)
            rind = ind+1
        if dr1*dr2 > 0:
            print('dammit, you have to do the latitude wrapping.')
            logger.debug('dr1 = %f, dr2 = %f, index = %d' % (dr1, dr2, ind))
            return rind, 1.0
        return (rind, np.abs(dr2)/(np.abs(dr1)+np.abs(dr2))+0.5)


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

    # need to find TRIANGLES (eventually TRAPEZOIDS) in which the z-component
    # of the vnormals is not shared across all vertices

    vnormals_per_triangle = mesh_front.vnormals[mesh_front.triangles]

    horizon_inds = []
    horizon_centers = np.array([])
    for i, vnormals_per_triangle in enumerate(vnormals_per_triangle):
        # if there is more than 1 sign in the array of z normals, then
        # the triangle is straggling the horizon and we need to store
        # the index.
        vnz = vnormals_per_triangle[:,2]
        if vnz.max() * vnz.min() < 0:
            # then we found the element that is crossing the horizon.  Now
            # let's check the triangle's znormal to see if we want this element
            # or its neighbor
            if mesh_front.centers[i] not in horizon_centers:
                horizon_inds.append(i)
                horizon_centers = np.append(horizon_centers, mesh_front.centers[i])

    # compute plane-of-sky polar coordinates (rho, theta) for the elements
    # in the front star
    rhos_front = np.sqrt((mesh_front.centers[:,0]-xs[i_front])**2 + (mesh_front.centers[:,1]-ys[i_front])**2)
    thetas_front = np.arctan2(mesh_front.centers[:,1]-ys[i_front], mesh_front.centers[:,0]-xs[i_front])
    # now get rhos and thetas for the back star but in the polar coordinate frame
    # of the front star
    rhos_back = np.sqrt((mesh_back.centers[:,0]-xs[i_front])**2 + (mesh_back.centers[:,1]-ys[i_front])**2)
    thetas_back = np.arctan2(mesh_back.centers[:,1]-ys[i_front], mesh_back.centers[:,0]-xs[i_front])

    if _EXPORT_HORIZON:
        tnormals_zsign = np.sign(mesh_front.tnormals[:,2])
        f = open('wd_horizon.horizon', 'w')
        f.write('#ind rho, theta, nz_sign\n')
        for ind in horizon_inds:
            # note here rhos and thetas need 0 not i_front because of the stupid way we did the list comprehension
            f.write("{} {} {} {}\n".format(ind, rhos_front[ind], thetas_front[ind], tnormals_zsign[ind]))
        f.close()

        f = open('wd_back_elements.txt', 'w')
        f.write('# ind, rho, theta\n')
        for ind in range(len(rhos_back)):
            f.write('%d %f %f\n' % (ind, rhos_back[ind], thetas_back[ind]))
        f.close()

    # do a fourier fit on the rhos and thetas of the horizon elements of the
    # front star
    horizon_fit, _ = cfit(fourier, thetas_front[horizon_inds], rhos_front[horizon_inds], np.ones(12))

    # filter out elements that are eclipsed based on the horizon fourier fit
    covered_mask = rhos_back < fourier(thetas_back, *horizon_fit)
    covered_indices = np.where(covered_mask)[0] # these are indices of covered elements
    covered_boundary_indices = np.sort(np.concatenate((covered_indices[0:1], covered_indices[np.where(covered_indices[1:]-covered_indices[:-1] != 1)], covered_indices[np.where(covered_indices[1:]-covered_indices[:-1] != 1)[0]+1], covered_indices[-1:])))

    # star with visibilities defined by horizon (normals) only
    visibilities, weights = only_horizon(meshes, xs, ys, zs)

    visibilities[comp_back][covered_indices] = 0.0

    # for all covered elements, let's compute the correction factor.  For WD
    # this number CAN be larger than 1.0
    for cbi in covered_boundary_indices:
        ind, cor = correction(rhos_back, thetas_back, cbi, *horizon_fit)
        visibilities[comp_back][ind] = cor

    return visibilities, None


def none(meshes, xs, ys, zs):
    """
    """
    return {comp_no: mesh.visibilities for comp_no, mesh in meshes.items()}, None

def only_horizon(meshes, xs, ys, zs):
    """
    Check all visible or partial triangles to see if they're behind the horizon,
    by checking the direction of the z-component of the normals (ie hidden if mu<0)
    """

    # if visibility == 0, it should remain 0
    # if visibility == 0.5, it should stay 0.5 if mu > 0 else it should become 0
    # if visibility == 1, it should stay 1 if mu > 0 else it should become 0

    # this can all by easily done by multiplying by int(mu>0) (1 if visible, 0 if hidden)

    return {comp_no: mesh.visibilities * (mesh.mus > 0).astype(int) for comp_no, mesh in meshes.items()}, None

def visible_ratio(meshes, xs, ys, zs):
    """
    TODO: add documentation

    this is the new eclipse detection method in libphoebe
    """

    centers_flat = meshes.get_column_flat('centers')
    vertices_flat = meshes.get_column_flat('vertices')
    triangles_flat = meshes.get_column_flat('triangles')  # should handle offset automatically
    normals_flat = meshes.get_column_flat('tnormals')

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
                                     horizon=_EXPORT_HORIZON)

    if _EXPORT_HORIZON:
        front_to_back_comp_nos = np.argsort(zs)[::-1]
        i_front = front_to_back_comp_nos[0]

        f = open('visible_ratio.horizon', 'w')
        f.write('#ignore rho theta\n')

        # need to find which of info['horizon'] is in the front
        zs = [np.median(vertices_flat[horizon][:,2]) for horizon in info['horizon']]
        h_front = np.array(zs).argmax()
        horizon_front = info['horizon'][h_front]

        for i in range(len(horizon_front)-1):
            # interpolate betweeen i and i+1 so we can see how the edge traces
            # in rho,theta space
            ind_this = horizon_front[i]
            ind_next = horizon_front[i+1]
            xs_edge = np.array([vertices_flat[ind_this][0], vertices_flat[ind_next][0]])
            ys_edge = np.array([vertices_flat[ind_this][1], vertices_flat[ind_next][1]])
            inds = xs_edge.argsort()
            xs_edge, ys_edge = xs_edge[inds], ys_edge[inds]
            x_sample = np.linspace(xs_edge[0], xs_edge[1], 101)
            y_sample = np.interp(x_sample, xs_edge, ys_edge)

            for x,y in zip(x_sample, y_sample):
                rho = np.sqrt((x-xs[i_front])**2 + (y-ys[i_front])**2)
                theta = np.arctan2(y-ys[i_front], x-xs[i_front])
                f.write('{} {} {}\n'.format(-1, rho, theta))


        f.close()

    visibilities = meshes.unpack_column_flat(info['tvisibilities'], computed_type='triangles')
    weights = meshes.unpack_column_flat(info['taweights'], computed_type='triangles')

    return visibilities, weights

def visible_partial(meshes, xs, ys, zs):
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

    visibilities = meshes.unpack_column_flat(visibilities)

    return visibilities, None



def graham(meshes, xs, ys, zs):
    """
    convex_graham
    """

    # once we delete or replace this with a version in libphoebe we can also
    # remove the ceclipse import statement, remove ceclipse from setup.py
    # and delete ceclipse.cpp in algorithms

    distance_factor = 1.0  # TODO: make this an option (what does it even do)?

    # first lets handle the horizon
    visibilities, weights = only_horizon(meshes, xs, ys, zs)

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


    return visibilities, None
