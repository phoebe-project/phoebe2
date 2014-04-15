#!/usr/bin/python

import os
import phoebe
from phoebe import universe
from phoebe.parameters import parameters, create, tools
from phoebe.backend import plotting
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.get_basic_logger()

exdir = os.path.dirname(__file__)

def test_pblums_l3(debug=False):
    """
    Pblum and l3 scaling: synthetic V380
    """
    return None
    parfile = os.path.join(exdir, 'V380_Cyg_circular.par')
    lcfile = os.path.join(exdir, 'mylc.lc')
    rv1file = os.path.join(exdir, 'myrv_comp1.rv')
    rv2file = os.path.join(exdir, 'myrv_comp2.rv')

    Apars,Bpars,orbitpars,globs = create.from_library(parfile, create_body=False)
    meshpars = parameters.ParameterSet(context='mesh:marching', delta=0.2)

    # load datasets
    obs_rv_a, pbdep_rv_a = phoebe.parse_rv(rv1file,
            passband='JOHNSON.V',
            columns=['time','rv','sigma'],
            components=[None,'myprimary','myprimary'],
            ref='myrv_comp1')

    obs_rv_b, pbdep_rv_b = phoebe.parse_rv(rv2file,
            passband='JOHNSON.V',
            columns=['time','rv','sigma'],
            components=[None,'mysecondary','mysecondary'],
            ref='myrv_comp2')

    obs_lc, pbdep_lc = phoebe.parse_lc(lcfile,
            passband='JOHNSON.V',
            columns=['time','flux','sigma'],
            components=[None,'V380Cyg','V380Cyg'],
            ref='mylc')
    
    
    # create bodies
    starA = universe.BinaryRocheStar(Apars, mesh=meshpars, orbit=orbitpars,
                                    pbdep=[pbdep_lc,pbdep_rv_a], obs=[obs_rv_a])
    starB = universe.BinaryRocheStar(Bpars, mesh=meshpars, orbit=orbitpars,
                                    pbdep=[pbdep_lc,pbdep_rv_b], obs=[obs_rv_b])
    system = universe.BinaryBag([starA,starB], orbit=orbitpars, label='V380Cyg',
                                position=globs,obs=[obs_lc])
    
    # compute
    #obs_lc.set_adjust('offset',True)
    #obs_lc.set_adjust('scale',True)
    tools.group([obs_rv_a, obs_rv_b], 'rv', scale=False, offset=True)
    compute_options = parameters.ParameterSet(context='compute')
    system.compute(params=compute_options, mpi=None)

    lcobs = system.get_obs(category='lc', ref='mylc').asarray()
    lcsyn = system.get_synthetic(category='lc', ref='mylc').asarray()

    rvobs1 = system[0].get_obs(category='rv', ref='myrv_comp1').asarray()
    rvsyn1 = system[0].get_synthetic(category='rv', ref='myrv_comp1').asarray()

    rvobs2 = system[1].get_obs(category='rv', ref='myrv_comp2').asarray()
    rvsyn2 = system[1].get_synthetic(category='rv', ref='myrv_comp2').asarray()

    if not debug:
        assert(np.mean((lcobs['flux']-(lcsyn['flux']*lcobs['scale'] - lcobs['offset']))**2/lcobs['sigma']**2)<0.00017)
        assert(np.mean((rvobs1['rv']-(rvsyn1['rv'] - rvobs1['offset']))**2/rvobs1['sigma']**2)<2.14)
        assert(np.mean((rvobs2['rv']-(rvsyn2['rv'] - rvobs2['offset']))**2/rvobs2['sigma']**2)<2.41)
    else:
        print rvobs1['rv']
        print rvsyn1['rv']
        print rvobs1['offset']
        diagnostics = np.mean((lcobs['flux']-(lcsyn['flux']*lcobs['scale'] - lcobs['offset']))**2/lcobs['sigma']**2),\
                      np.mean((rvobs1['rv']-(rvsyn1['rv'] - rvobs1['offset']))**2/rvobs1['sigma']**2),\
                      np.mean((rvobs2['rv']-(rvsyn2['rv'] - rvobs2['offset']))**2/rvobs2['sigma']**2)
        return system, diagnostics



if __name__=='__main__':
    
    system, diag = test_pblums_l3(debug=True)
    print diag
    starA = system[0]
    starB = system[1]
    # plot
    plt.cla()
    plotting.plot_lcobs(system, ref='mylc', color='k', marker='o', errorbars=False)
    plotting.plot_lcsyn(system, ref='mylc', color='r', linewidth=2)
    plt.show()

    plt.cla()
    plotting.plot_rvobs(starA, ref='myrv_comp1', color='b', marker='o', linestyle='', errorbars=False)
    plotting.plot_rvobs(starB, ref='myrv_comp2', color='r', marker='o', linestyle='', errorbars=False)
    plotting.plot_rvsyn(starA, ref='myrv_comp1', color='b')
    plotting.plot_rvsyn(starB, ref='myrv_comp2', color='r')
    plt.show()
