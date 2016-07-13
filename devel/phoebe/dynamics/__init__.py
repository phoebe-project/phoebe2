
import nbody as nbody
import keplerian as keplerian
from phoebe import u


# def dynamics_at_i(xs, ys, zs, vxs, vys, vzs, ethetas=None, elongans=None, eincls=None, i=0):
#     xi, yi, zi = [x[i].to(u.solRad) for x in xs], [y[i].to(u.solRad) for y in ys], [z[i].to(u.solRad) for z in zs]
#     vxi, vyi, vzi = [vx[i].to(u.solRad/u.d) for vx in vxs], [vy[i].to(u.solRad/u.d) for vy in vys], [vz[i].to(u.solRad/u.d) for vz in vzs]
#     if ethetas is not None:
#         ethetai, elongani, eincli = [etheta[i].to(u.rad) for etheta in ethetas], [elongan[i].to(u.rad) for elongan in elongans], [eincl[i].to(u.rad) for eincl in eincls]
#     else:
#         ethetai, elongani, eincli = None, None, None

#     return xi, yi, zi, vxi, vyi, vzi, ethetai, elongani, eincli

def at_i(array, i=0):
    return [a[i] for a in array]


def dynamics_at_i(xs, ys, zs, vxs, vys, vzs, ethetas=None, elongans=None, eincls=None, i=0):
    # xi, yi, zi = [x[i] for x in xs], [y[i] for y in ys], [z[i] for z in zs]
    xi, yi, zi = at_i(xs, i), at_i(ys, i), at_i(zs, i)
    # vxi, vyi, vzi = [vx[i] for vx in vxs], [vy[i] for vy in vys], [vz[i] for vz in vzs]
    vxi, vyi, vzi = at_i(vxs, i), at_i(vys, i), at_i(vzs, i)
    if ethetas is not None:
        # ethetai, elongani, eincli = [etheta[i] for etheta in ethetas], [elongan[i] for elongan in elongans], [eincl[i] for eincl in eincls]
        ethetai, elongani, eincli = at_i(ethetas, i), at_i(elongans, i), at_i(eincls, i)
    else:
        ethetai, elongani, eincli = None, None, None

    return xi, yi, zi, vxi, vyi, vzi, ethetai, elongani, eincli
