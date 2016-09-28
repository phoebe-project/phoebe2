"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

phoebe.devel_on()

def _phoebe_v_legacy_lc_protomesh(b, gridsize=50, plot=False):
    """
    """

    b = phoebe.Bundle.default_binary()

    b.add_dataset('lc', times=[0], dataset='lc01')

    b.add_compute('legacy', compute='phoebe1')
    b.add_compute('phoebe', compute='phoebe2', subdiv_num=0)

    b.set_value_all('mesh_method', 'wd')
    b.set_value_all('eclipse_method', 'graham')
    b.set_value_all('gridsize', gridsize)

    # TODO: make these options and test over various values for the intensity
    b.set_value_all('ld_func', 'logarithmic')
    b.set_value_all('ld_coeffs', [0,0])
    # TODO: also compare phoebe1:kurucz to phoebe:extern_atmx
    b.set_value_all('atm@phoebe1', 'blackbody')
    b.set_value_all('atm@phoebe2', 'extern_planckint')

    b.run_compute('phoebe1', model='phoebe1model', protomesh=True, pbmesh=True, refl_num=0)
    b.run_compute('phoebe2', model='phoebe2model', protomesh=True, pbmesh=True, reflection_method='none')


    compares = []
    # compares += [{'qualifier': 'xs', 'dataset': 'protomesh', 'atol': 1e-10}]
    # compares += [{'qualifier': 'ys', 'dataset': 'protomesh', 'atol': 1e-12}]
    # compares += [{'qualifier': 'zs', 'dataset': 'protomesh', 'atol': 1e-11}]
    # compares += [{'qualifier': 'rs', 'dataset': 'protomesh', 'atol': 1e-10}]
    # compares += [{'qualifier': 'nxs', 'dataset': 'protomesh', 'atol': 1e-7}]
    # compares += [{'qualifier': 'nys', 'dataset': 'protomesh', 'atol': 1e-9}]
    # compares += [{'qualifier': 'nzs', 'dataset': 'protomesh', 'atol': 1e-8}]
    # compares += [{'qualifier': 'cosbetas', 'dataset': 'protomesh', 'atol': 1e-14}]

    compares += [{'qualifier': 'loggs', 'dataset': 'protomesh', 'atol': 2e-4}]
    compares += [{'qualifier': 'teffs', 'dataset': 'protomesh', 'atol': 1e-6}]

    compares += [{'qualifier': 'abs_normal_intensities', 'dataset': 'lc01', 'atol': 1e5}] # NOTE: these values are of order 1E14


    for c in compares:
        qualifier = c['qualifier']
        dataset = c.get('dataset', 'protomesh')
        for component in b.hierarchy.get_stars():


            phoebe1_val = b.get_value(section='model', model='phoebe1model', component=component, dataset=dataset, qualifier=qualifier)
            phoebe2_val = b.get_value(section='model', model='phoebe2model', component=component, dataset=dataset, qualifier=qualifier)


            if component=='secondary':
                # TODO: this logic should /REALLY/ be moved into the legacy backend wrapper
                if qualifier in ['xs']:
                    # the secondary star from phoebe 1 is at (d=a=1, 0, 0)
                    phoebe2_val -= 1
                if qualifier in ['xs', 'ys', 'nxs', 'nys']:
                    # the secondary star from phoebe1 is rotated about the z-axis
                    phoebe2_val *= -1


            # TODO: handle the hemispheres correctly in the legacy backend and remove this [::8] stuff (also below in plotting)
            if dataset=='protomesh':
                phoebe1_val = phoebe1_val[::8]
                phoebe2_val = phoebe2_val[::8]


            print "{}@{}@{} max diff: {}".format(qualifier, component, dataset, max(np.abs(phoebe1_val-phoebe2_val)))

            if plot:
                x = b.get_value(section='model', model='phoebe2model', component=component, dataset='protomesh', qualifier='xs')

                if dataset=='protomesh':
                    x = x[::8]

                fig, (ax1, ax2) = plt.subplots(1,2)

                ax1.plot(x, phoebe1_val, 'bo')
                ax1.plot(x, phoebe2_val, 'r.')

                ax2.plot(x, phoebe1_val-phoebe2_val, 'k.')

                ax1.set_xlabel('{}@{} (phoebe2)'.format('x', component))
                ax2.set_xlabel('{}@{} (phoebe2)'.format('x', component))

                ax1.set_ylabel('{}@{} (blue=phoebe1, red=phoebe2)'.format(qualifier, component))
                ax2.set_ylabel('{}@{} (phoebe1-phoebe2)'.format(qualifier, component))

                plt.show()
                plt.close(fig)

            assert(np.allclose(phoebe1_val, phoebe2_val, atol=c.get('atol', 1e-7), rtol=c.get('rtol', 0.0)))




def test_binary(plot=False):
    """
    """

    # TODO: try an eccentric orbit over multiple phases (will need to wait for non-protomesh support from the legacy wrapper)
    # TODO: once ps.copy is implemented, just send b.copy() to each of these

    b = phoebe.Bundle.default_binary()
    _phoebe_v_legacy_lc_protomesh(b, plot=plot)

if __name__ == '__main__':
    logger = phoebe.logger()


    test_binary(plot=True)


