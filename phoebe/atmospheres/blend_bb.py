import phoebe
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def edge(new_axes, new_table):
    edge = np.nan*np.ones((len(new_axes[0]), len(new_axes[1]), len(new_axes[2]), 1))

    for Ti in range(len(new_axes[0])):
        for Li in range(len(new_axes[1])):
            for Mi in range(len(new_axes[2])):
                if np.isnan(new_table[Ti, Li, Mi, 0]):
                    continue
                
                num_directions = 0
                extrapolated_value = 0.0

                if (Mi+1 < len(new_axes[2]) and np.isnan(new_table[Ti, Li, Mi+1, 0])) or (Mi > 1 and np.isnan(new_table[Ti, Li, Mi-1, 0])):
                    edge[Ti, Li, Mi, 0] = new_table[Ti, Li, Mi, 0]
                if (Li+1 < len(new_axes[1]) and np.isnan(new_table[Ti, Li+1, Mi, 0])) or (Li > 1 and np.isnan(new_table[Ti, Li-1, Mi, 0])):
                    edge[Ti, Li, Mi, 0] = new_table[Ti, Li, Mi, 0]
                if (Ti+1 < len(new_axes[0]) and np.isnan(new_table[Ti+1, Li, Mi, 0])) or (Ti > 1 and np.isnan(new_table[Ti-1, Li, Mi, 0])):
                    edge[Ti, Li, Mi, 0] = new_table[Ti, Li, Mi, 0]

    return edge

def extrapolate(new_axes, axes, table):
    if new_axes is None:
        new_axes = []

        # add extrapolation knots:
        for i, axis in enumerate(axes):
            new_axes.append(np.insert(axis, (0, len(axis)), (axis[0]-(axis[1]-axis[0]), axis[len(axis)-1]+(axis[len(axis)-1]-axis[len(axis)-2]))))

    # make sure that new_axes contain axes:
    for i in range(len(axes)):
        if axes[i].tostring() not in new_axes[i].tostring():
            print('axes must be contained in new_axes; aborting.')
            return None

    new_table = np.nan*np.ones((len(new_axes[0]), len(new_axes[1]), len(new_axes[2]), 1))

    if new_axes is None:
        new_table[1:-1,1:-1,1:-1] = table
    else:
        # find an overlap between axes and new_axes:
        Ti, Tl = new_axes[0].tostring().index(axes[0].tostring())/new_axes[0].itemsize, len(axes[0])
        Li, Ll = new_axes[1].tostring().index(axes[1].tostring())/new_axes[1].itemsize, len(axes[1])
        Mi, Ml = new_axes[2].tostring().index(axes[2].tostring())/new_axes[2].itemsize, len(axes[2])

        new_table[Ti:Ti+Tl,Li:Li+Ll,Mi:Mi+Ml] = table

    extrapolant = np.nan*np.ones_like(new_table)

    for Ti in range(len(new_axes[0])):
        for Li in range(len(new_axes[1])):
            for Mi in range(len(new_axes[2])):
                if not np.isnan(new_table[Ti, Li, Mi, 0]):
                    continue
                
                num_directions = 0
                extrapolated_value = 0.0

                if Mi+2 < len(new_axes[2]) and not np.isnan(new_table[Ti, Li, Mi+1, 0]) and not np.isnan(new_table[Ti, Li, Mi+2, 0]):
                    extrapolated_value += 2*new_table[Ti, Li, Mi+1,0]-new_table[Ti, Li, Mi+2, 0]
                    # print('M[%d,%d,%d] is right-defined in metallicity, extrap=%f' % (Ti, Li, Mi, extrapolated_value))
                    num_directions += 1

                if Mi > 2 and not np.isnan(new_table[Ti, Li, Mi-1, 0]) and not np.isnan(new_table[Ti, Li, Mi-2, 0]):
                    extrapolated_value += 2*new_table[Ti, Li, Mi-1,0]-new_table[Ti, Li, Mi-2, 0]
                    # print('M[%d,%d,%d] is right-defined in metallicity, extrap=%f' % (Ti, Li, Mi, extrapolated_value))
                    num_directions += 1

                if Li+2 < len(new_axes[1]) and not np.isnan(new_table[Ti, Li+1, Mi, 0]) and not np.isnan(new_table[Ti, Li+2, Mi, 0]):
                    extrapolated_value += 2*new_table[Ti, Li+1, Mi,0]-new_table[Ti, Li+2, Mi, 0]
                    # print('M[%d,%d,%d] is right-defined in log(g), extrap=%f' % (Ti, Li, Mi, extrapolated_value))
                    num_directions += 1

                if Li > 2 and not np.isnan(new_table[Ti, Li-1, Mi, 0]) and not np.isnan(new_table[Ti, Li-2, Mi, 0]):
                    extrapolated_value += 2*new_table[Ti, Li-1, Mi,0]-new_table[Ti, Li-2, Mi, 0]
                    # print('M[%d,%d,%d] is left-defined in log(g), extrap=%f' % (Ti, Li, Mi, extrapolated_value))
                    num_directions += 1

                if Ti+2 < len(new_axes[0]) and not np.isnan(new_table[Ti+1, Li, Mi, 0]) and not np.isnan(new_table[Ti+2, Li, Mi, 0]):
                    extrapolated_value += 2*new_table[Ti+1, Li, Mi,0]-new_table[Ti+2, Li, Mi, 0]
                    # print('M[%d,%d,%d] is right-defined in temperature, extrap=%f' % (Ti, Li, Mi, extrapolated_value))
                    num_directions += 1

                if Ti > 2 and not np.isnan(new_table[Ti-1, Li, Mi, 0]) and not np.isnan(new_table[Ti-2, Li, Mi, 0]):
                    extrapolated_value += 2*new_table[Ti-1, Li, Mi,0]-new_table[Ti-2, Li, Mi, 0]
                    # print('M[%d,%d,%d] is left-defined in temperature, extrap=%f' % (Ti, Li, Mi, extrapolated_value))
                    num_directions += 1

                if num_directions == 0:
                    continue

                extrapolant[Ti, Li, Mi, 0] = extrapolated_value/num_directions

    return (new_table, extrapolant)


pb = phoebe.get_passband('Kepler:mean')

ck_axes = pb._ck2004_axes
ck_ints = pb._ck2004_energy_grid

new_axes = (
    np.concatenate((np.arange(250., 3251, 250), ck_axes[0], np.arange(55000., 500001, 5000))),
    np.concatenate((ck_axes[1], np.arange(5.5, 10.1, 0.5))),
    ck_axes[2]
)

new_table, extrapolant = extrapolate(new_axes, ck_axes, ck_ints)

bb_table = np.empty_like(new_table)
for Ti, T in enumerate(new_axes[0]):
    for Li in range(len(new_axes[1])):
        for Mi in range(len(new_axes[2])):
            bb_table[Ti, Li, Mi, 0] = pb._log10_Inorm_bb_energy(T)
            # bb_table[Ti, Li, Mi, 0] = np.log10(pb._bb_intensity(T, photon_weighted=False))

# blend the edge:
ck_edge = edge(new_axes, new_table)
blend = ck_edge * 0.5 + bb_table * 0.5

# blend the extrapolated edge:
blend_e = extrapolant * 0.25 + bb_table * 0.75

# peal the edge:
np.nan_to_num(ck_edge, copy=False)
pealed_table = new_table - ck_edge
pealed_table[pealed_table == 0] = np.nan

# blend the pealed edge:
pealed_edge = edge(new_axes, pealed_table)
blend_p = pealed_edge * 0.75 + bb_table * 0.25

# np.nan_to_num(inner_edge, copy=False)

new_table[~np.isnan(blend)] = blend[~np.isnan(blend)]
new_table[~np.isnan(blend_p)] = blend_p[~np.isnan(blend_p)]
new_table[~np.isnan(blend_e)] = blend_e[~np.isnan(blend_e)]

# finally, adopt blackbody everywhere else:
new_table[np.isnan(new_table)] = bb_table[np.isnan(new_table)]

# pealed_table = pealed_table - inner_edge
# pealed_table[pealed_table == 0] = np.nan

plt.imshow(new_table[:,:,5,0].T)
plt.show()

pb._blended_axes = new_axes
pb._blended_energy_grid = new_table
pb.content += ['blended']
pb.save('kepler_blended.pb')

exit()



inner = 0.75* + 0.25*blackbody_grid
ck_edge = 0.5*new_edge + 0.5*blackbody_grid
outer = 0.25*extrapolant + 0.75*blackbody_grid

for Ti in range(len(new_axes[0])):
    for Li in range(len(new_axes[1])):
        for Mi in range(len(new_axes[2])):
            if np.isnan(blend[Ti,Li,Mi,0]):
                continue
            new_energy_grid[Ti,Li,Mi,0] = blend[Ti,Li,Mi,0]

exit()

plt.xlabel('Teff [K]')
plt.ylabel('logg [cgs]')

leg = False
for i, T in enumerate(new_axes[0]):
    for j, l in enumerate(new_axes[1]):
        if np.isnan(new_energy_grid[i,j,5,0]):
            continue
        if not leg:
            plt.plot(T, l, 'bs', label='extended matrix elements')
            leg = True
        else:
            plt.plot(T, l, 'bs')

leg = False
for i, T in enumerate(new_axes[0]):
    for j, l in enumerate(new_axes[1]):
        if np.isnan(extrapolant[i,j,5,0]):
            continue
        if not leg:
            plt.plot(T, l-0.05, 'gs', label='extrapolated elements')
            leg = True
        else:
            plt.plot(T, l-0.05, 'gs')

leg = False
for i, T in enumerate(ck_axes[0]):
    for j, l in enumerate(ck_axes[1]):
        if np.isnan(ck_ints[i,j,4,0]):
            continue
        if not leg:
            plt.plot(T, l-0.1, 'rs', label='original ck2004 elements')
            leg = True
        else:
            plt.plot(T, l-0.1, 'rs')

plt.legend(loc='lower right')
plt.show()

# np.nan_to_num(new_energy_grid, copy=False)

# new_energy_grid += 0.25*extrapolant + 0.75*blackbody_grid

plt.imshow(blackbody_grid[:,:,5,0].T)
plt.show()

# for i, L in enumerate(new_axes[1]):
#     for j, M in enumerate(new_axes[2]):
#         if np.isnan(new_energy_grid[15,i,j,0]):
#             continue
#         plt.plot(L, M, 'bs')

# for i, L in enumerate(ck_axes[1]):
#     for j, M in enumerate(ck_axes[2]):
#         if np.isnan(ck_ints[14,i,j,0]):
#             continue
#         plt.plot(L, M-0.1, 'rs')
# plt.show()