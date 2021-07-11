"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt
import os

def _phoebe_v_legacy_lc_protomesh(b, gridsize=10, plot=False, gen_comp=False):
    """
    """

    b.add_dataset('lc', times=[0], dataset='lc01')
    b.add_dataset('mesh', include_times='lc01', dataset='mesh01', columns=['abs_normal_intensities@*', 'loggs', 'teffs', 'xs'])

    b.add_compute('legacy', compute='phoebe1')
    b.add_compute('phoebe', compute='phoebe2')

    b.set_value_all('mesh_method', 'wd')
    b.set_value_all('eclipse_method', 'graham')
    b.set_value_all('gridsize', gridsize)

    # TODO: make these options and test over various values for the intensity
    b.set_value_all('ld_mode', 'manual')
    b.set_value_all('ld_func', 'linear')
    b.set_value_all('ld_coeffs', [0.])
    # TODO: also compare phoebe1:kurucz to phoebe:extern_atmx
    b.set_value_all('atm', 'extern_planckint')

    #turn off albedos (legacy requirement)
    b.set_value_all('irrad_frac_refl_bol',  0.0)

    if gen_comp:
        b.run_compute('phoebe1', model='phoebe1model', refl_num=0)
        b.filter(model='phoebe1model').save('test_mesh.comp.model')
    else:
        b.import_model(os.path.join(os.path.dirname(__file__), 'test_mesh.comp.model'), model='phoebe1model')
    b.run_compute('phoebe2', model='phoebe2model', irrad_method='none')


    compares = []
    # compares += [{'qualifier': 'us', 'dataset': 'mesh01', 'atol': 1e-10}]
    # compares += [{'qualifier': 'vs', 'dataset': 'mesh01', 'atol': 1e-12}]
    # compares += [{'qualifier': 'ws', 'dataset': 'mesh01', 'atol': 1e-11}]
    # compares += [{'qualifier': 'rs', 'dataset': 'mesh01', 'atol': 1e-10}]
    # compares += [{'qualifier': 'nus', 'dataset': 'mesh01', 'atol': 1e-7}]
    # compares += [{'qualifier': 'nvs', 'dataset': 'mesh01', 'atol': 1e-9}]
    # compares += [{'qualifier': 'nws', 'dataset': 'mesh01', 'atol': 1e-8}]
    # compares += [{'qualifier': 'cosbetas', 'dataset': 'mesh01', 'atol': 1e-14}]

    compares += [{'qualifier': 'xs', 'dataset': 'mesh01', 'atol': 1e-4}]
    compares += [{'qualifier': 'loggs', 'dataset': 'mesh01', 'atol': 2e-4}]
    compares += [{'qualifier': 'teffs', 'dataset': 'mesh01', 'atol': 1e-5}]

    compares += [{'qualifier': 'abs_normal_intensities', 'dataset': 'lc01', 'atol': 0, 'rtol': 1e-8}] # NOTE: these values are of order 1E14


    for c in compares:
        qualifier = c['qualifier']
        dataset = c.get('dataset', 'protomesh')
        for component in b.hierarchy.get_stars():


            phoebe1_val = b.get_value(section='model', model='phoebe1model', component=component, dataset=dataset, qualifier=qualifier)
            phoebe2_val = b.get_value(section='model', model='phoebe2model', component=component, dataset=dataset, qualifier=qualifier)


            # phoebe2 wd-style mesh duplicates each trapezoid into two
            # triangles, so we only need every other.  It also handles
            # duplicating quadrants per-element, whereas the phoebe1
            # wrapper duplicates per-quadrant.  So we also need to reshape.
            # TODO: move this into the order that these are actually
            # exposed by PHOEBE to the user
            phoebe2_val = phoebe2_val[::2].reshape(-1,4).flatten(order='F')

            if component=='secondary':
                # TODO: this logic should /REALLY/ be moved into the legacy backend wrapper
                if qualifier in ['xs']:
                    # the secondary star from phoebe 1 is at (d=1, 0, 0)
                    phoebe2_val -= 1
                if qualifier in ['xs', 'ys', 'nxs', 'nys']:
                    # the secondary star from phoebe1 is rotated about the z-axis
                    phoebe2_val *= -1


            if plot:
                print("{}@{}@{} max diff: {}".format(qualifier, component, dataset, max(np.abs(phoebe1_val-phoebe2_val))))

            if plot:
                x1 = b.get_value(section='model', model='phoebe1model', component=component, dataset='mesh01', qualifier='xs')
                # x2 = b.get_value(section='model', model='phoebe2model', component=component, dataset='mesh01', qualifier='xs')[::2]

                fig, (ax1, ax2) = plt.subplots(1,2)
                ax1.plot(x1, phoebe1_val, 'bo')
                ax1.plot(x1, phoebe2_val, 'r.')

                ax2.plot(x1, phoebe1_val-phoebe2_val, 'k.')

                ax1.set_xlabel('{}@{} (phoebe1)'.format('x', component))
                ax2.set_xlabel('{}@{} (phoebe1)'.format('x', component))

                ax1.set_ylabel('{}@{} (blue=phoebe1, red=phoebe2)'.format(qualifier, component))
                ax2.set_ylabel('{}@{} (phoebe1-phoebe2)'.format(qualifier, component))

                plt.show()
                plt.close(fig)

            assert(np.allclose(phoebe1_val, phoebe2_val, atol=c.get('atol', 1e-7), rtol=c.get('rtol', 0.0)))




def test_binary(plot=False, gen_comp=False):
    """
    """

    phoebe.devel_on() # required for wd meshing

    # TODO: try an eccentric orbit over multiple phases (will need to wait for non-protomesh support from the legacy wrapper)
    # TODO: once ps.copy is implemented, just send b.copy() to each of these

    b = phoebe.Bundle.default_binary()
    _phoebe_v_legacy_lc_protomesh(b, plot=plot, gen_comp=gen_comp)

    phoebe.devel_off() # reset for future tests

if __name__ == '__main__':
    logger = phoebe.logger('debug')


    test_binary(plot=True, gen_comp=True)
