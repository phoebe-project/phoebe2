"""
Compare Phoebe Legacy output with Phoebe 2.0.

Usage:

To do a full comparison of light curves and radial velocity curves:

    $:> python compare2.0.py detached_2
    
To make images of the system at a specific time:
    
    $:> python compare2.0.py detached_2 1.75
    
"""
import sys
import os
import numpy as np
import phoebe
from phoebe.io import parsers
from phoebe.parameters import datasets
from phoebe.parameters import tools
from matplotlib import pyplot as plt

logger = phoebe.get_basic_logger()
#logger = phoebe.get_basic_logger(clevel=None,filename='detached_1.log')

def compare(name):
    
    # Parse the Legacy file
    system = parsers.legacy_to_phoebe("{}.phoebe".format(name), mesh='marching',
                                    create_body=True)
    print(system.list(summary='long'))
    
    # Override the mesh density, we don't want to wait forever
    system[0].params['mesh']['delta'] = 0.15
    system[1].params['mesh']['delta'] = 0.15
    
    # Get the references
    ref_lc = system.get_refs(category='lc')[0]
    ref_rv1 = system.get_refs(category='rv')[0]
    ref_rv2 = system.get_refs(category='rv')[1]

    # Add Phoebe 1.0 results as observations, but only if they exist
    lc_file = "{}.lc.1.0.data".format(name)

    system_obs = []

    if os.path.isfile(lc_file):
        lcobs, lcdep = datasets.parse_lc(lc_file, ref=ref_lc)
        system_obs.append(lcobs)

    system.add_obs(system_obs)

    # Compute Phoebe 2.0 results
    phoebe.compute(system, eclipse_alg='convex', mpi=True)


    results = []

    # Make comparison plots

    try:
        diffs = plot_lc(system, ref_lc)
        results.append([np.median(diffs), diffs.std(), diffs.min(), diffs.max()])
    except:
        pass
    
    try:
        diffs = plot_rv(system, 0, ref_rv1)
        results.append([np.median(diffs), diffs.std(), diffs.min(), diffs.max()])
    except:
        pass
    
    try:
        diffs = plot_rv(system, 1, ref_rv2)
        results.append([np.median(diffs), diffs.std(), diffs.min(), diffs.max()])
    except:
        pass
    
    
    print(("{:20s}"*4).format('MEDIAN','STDDEV','MIN',"MAX"))
    for result in results:
        print(("{:20.8f}"*4).format(*result))
    
    return results


def snapshot(name, time):
    
    # Parse the Legacy file
    system = parsers.legacy_to_phoebe("{}.phoebe".format(name), mesh='marching',
                                    create_body=True)
    print(system.list(summary='long'))
    
    # Override the mesh density, we don't want to wait forever
    system[0].params['mesh']['delta'] = 0.15
    system[1].params['mesh']['delta'] = 0.15
    
    # Get the references
    ref_lc = system.get_refs(category='lc')[0]
    ref_rv1 = system.get_refs(category='rv')[0]
    ref_rv2 = system.get_refs(category='rv')[1]

    # Add Phoebe 1.0 results as observations, but only if they exist
    lc_file = "{}.lc.1.0.data".format(name)

    system_obs = []

    if os.path.isfile(lc_file):
        lcobs, lcdep = datasets.parse_lc(lc_file, ref=ref_lc)
        system_obs.append(lcobs)

    system.add_obs(system_obs)

    # Compute Phoebe 2.0 results
    params = phoebe.ParameterSet(context='compute')
    phoebe.observatory.extract_times_and_refs(system, params)
    for i, j, k in zip(params['time'], params['refs'], params['types']):
        if np.abs(i-time)>0.001:
            new_list = params['time']
            new_list.remove(i)
            params['time'] = new_list
            new_list = params['refs']
            new_list.remove(j)
            params['refs'] = new_list
            new_list = params['types']
            new_list.remove(k)
            params['types'] = new_list
    
    phoebe.compute(system, params=params, eclipse_alg='convex')
    
    ps, lc, rv = parsers.phoebe_to_wd(system)
    lc2 = lc.copy()
    lc2['indep'] = [1.31]
    curve, params = phoebe.wd.lc(ps, request='image', light_curve=lc2, rv_curve=rv)
    lc['indep'] = lc['indep'][np.abs(lc['indep']-time)<=0.001]
    curve, params = phoebe.wd.lc(ps, request='image', light_curve=lc, rv_curve=rv)
    
    plt.figure()
    ax = plt.subplot(221)
    plt.title("Projected intensity")
    system.plot2D(ref=0, ax=ax)
    ax = plt.subplot(222)
    plt.title("WD comparison")
    system.plot2D(ref=0, ax=ax)
    plt.plot(curve['y']*ps['sma'], curve['z']*ps['sma'], 'ro', mec='r', ms=2)
    ax = plt.subplot(223)
    plt.title("Effective temperature")
    out = system.plot2D(ref=0, select='teff', ax=ax)
    plt.colorbar(out[2])
    ax = plt.subplot(224)
    plt.title("Radial velocity")
    out = system.plot2D(ref=0, select='rv', ax=ax)
    plt.colorbar(out[2])
    
    tools.summarize(system)
    
    plt.show()
    
    from mayavi import mlab
    mlab.figure()
    system.plot3D()
    mlab.figure()
    system.plot3D(velos=True)
    mlab.figure()
    system.plot3D(normals=True)
    
    mlab.show()



def plot_lc(system, ref_lc):
    syn = system.get_synthetic(category='lc', ref=ref_lc).asarray()
    obs = system.get_obs(category='lc', ref=ref_lc)
    
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    plt.title(ref_lc)
    ax.set_xlim(syn['time'].min(), syn['time'].max())
    ax.set_xlabel('Phase')
    ax.set_ylabel('Flux')
    plt.plot(obs['time'], obs['flux'], 'b-', lw=2, label='Old')
    plt.plot(syn['time'], syn['flux'], 'r--', lw=2, label='New')
    plt.legend(loc='best').get_frame().set_alpha(0.5)
    
    pd = fig.add_subplot(2, 1, 2)
    ax.set_xlim(syn['time'].min(), syn['time'].max())
    
    pd.set_xlabel('Phase')
    pd.set_ylabel('Differences (new-old)')
    diffs = syn['flux']-obs['flux']
    plt.plot(obs['time'], diffs, 'b-', lw=2)
    return diffs


def plot_rv(system, comp, ref_rv):
    syn = system[comp].get_synthetic(category='rv', ref=ref_rv).asarray()
    obs = system[comp].get_obs(category='rv', ref=ref_rv)
    
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    plt.title(ref_rv)
    ax.set_xlim(syn['time'].min(), syn['time'].max())
    ax.set_xlabel('Phase')
    ax.set_ylabel('RV')
    plt.plot(obs['time'], obs['rv'], 'b-', lw=2, label='Old')
    plt.plot(syn['time'], syn.get_value('rv', 'km/s'), 'r--', lw=2, label='New')
    plt.legend(loc='best').get_frame().set_alpha(0.5)
    
    pd = fig.add_subplot(2, 1, 2)
    ax.set_xlim(syn['time'].min(), syn['time'].max())
    
    pd.set_xlabel('Phase')
    pd.set_ylabel('Differences (new-old)')
    diffs = syn.get_value('rv', 'km/s')-obs['rv']
    plt.plot(obs['time'], diffs, 'b-', lw=2)
    
    return diffs
    

if __name__=="__main__":
    if sys.argv[1:]:
        name = sys.argv[1]
    else:
        name = "detached_1"
        
    if sys.argv[2:]:
        snapshot(name, float(sys.argv[2]))
    else:
        compare(name)
    
    plt.show()
    

