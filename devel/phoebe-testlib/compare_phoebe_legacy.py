"""
Compare Phoebe Legacy light curve with Phoebe 2.0 light curve.

Usage:

$:> python compare_phoebe_legacy 4544587

It will create a directory phoebe2.0_4544587 if it does not exist. The phoebe
input file in directory '4544587' will be read and converted to a Phoebe 2.0
BodyBag. If there are no light curve files computed with Phoebe 2.0 in the
directory phoebe2.0_4544587, they will be computed and saved there. Otherwise,
they will just be loaded and plotted.

The meshes at critical phases are computed and stored, if they do not exist
yet. They will also not be recomputed if they already exist.

The WD lcin file will also be read and passed to pywd to compute the light
curve. In principle, this should be the exact same light curve as computed
with Phoebe Legacy, though the WD version is slightly different. Pywd uses
the 2003 version of WD.

If it takes a *very* long time to compute stuff, there are three ways of
making it run faster:

1. Choose circular orbits
2. Choose a lower mesh density
3. Switch off reflection effects.

If you want to profile an example, do:

$:> python -m cProfile -o myprofile.cprof compare_phoebe_legacy 4544587

And to analyse (if you have runSnakeRun installed):

$:> runsnake myprofile.cprof

Beware that if you profile the code, it will not be run through MPI, and so
it can take a *long* time to run it!

"""
from matplotlib import pyplot as plt
import socket
import os
import sys
import glob
import numpy as np
import multiprocessing
import phoebe
from phoebe import wd
from phoebe.backend import universe
from phoebe.dynamics import keplerorbit
from phoebe.algorithms import eclipse
from phoebe import kelly
from pyphoebe.parameters import create

logger = phoebe.get_basic_logger()

def compare(test_case,delta=0.1,recompute=False,mesh='marching',
            conserve_volume='periastron'):
    """
    Main function to compare Phoebe 2.0 vs Phoebe Legacy vs WD.
    """
    #-- figure out which critical phase is the reference for conservation of
    #   volume
    cvol_index = ['periastron','sup_conj','inf_conj','asc_node','desc_node'].index(conserve_volume)
    
    #========= FILES and INPUTS =============================================
    #--------- PHOEBE LEGACY ------------------------------------------------
    #-- collect all the files we need, and define the names of the files we
    #   wish to create:
    #   First, the reference files from the LEGACY code, provided by Kelly:
    legacy_input_file = glob.glob(os.path.join(test_case,'*.phoebe'))[0]
    legacy_basename   = legacy_input_file.rstrip('-test.phoebe')
    legacy_lc_file    = legacy_basename + '-time.dat'
    legacy_crit_times,legacy_crit_files = get_meshes(legacy_basename)
    #   read Phoebe Legacy input using Kelly's parser:
    system = kelly.legacy_to_phoebe(legacy_input_file,create_body=True,mesh=mesh)
    #   change from definition of superior conjunction to periastron passage
    create.from_supconj_to_perpass(system[0].params['orbit'])    
    #-- Load the contents of the Phoebe Legacy LC file, we'll need this later
    #   to compare the results.
    otime,oflux = np.loadtxt(legacy_lc_file).T[:2]
    
    #--------- PHOEBE 2.0 ---------------------------------------------------
    #-- We will save the results from Phoebe 2.0 calculations so that we don't
    #   need to recompute all stuff if we only want to make plots:
    #   The files to be created from PHOEBE2.0 are
    phoebe_basename = 'phoebe2.0_' + legacy_basename
    phoebe_lc_file = phoebe_basename + '-test.lc'
    phoebe_lc1_file = phoebe_basename + '-test.lc1'
    phoebe_lc2_file = phoebe_basename + '-test.lc2'
    phoebe_crit_files = [phoebe_basename + os.path.basename(crit_file) for crit_file in legacy_crit_files]
    phoebe_out_file = phoebe_basename + '.pars'
    
    #   Check if our critical times are the same as those from Legacy
    t0 = system[0].params['orbit']['t0']
    P = system[0].params['orbit']['period']
    per0 = system[0].params['orbit'].request_value('per0','rad')
    ecc = system[0].params['orbit'].request_value('ecc')
    phshift = system[0].params['orbit'].request_value('phshift')
    phoebe_crit_times = keplerorbit.calculate_critical_phases(per0,ecc)*P + t0
    #-- check if Phoebe 2.0 critical times are equal to Phoebe Legacy's
    print("Period={}, t0={}, phshift={}".format(P,t0,phshift))
    if not np.all(legacy_crit_times==phoebe_crit_times):
        for ff,lt,pt in zip(legacy_crit_files,legacy_crit_times,phoebe_crit_times):
            print '{:40s}: {:.6} <-> {:.6} ({:.4}, or {:.4} phase units)'.format(ff,lt,pt,lt-pt,np.mod(lt-pt,P)/P)
        logger.error("There is something wrong with the times of critical phases...")
    
    #-- check if files and directories exist to write in:
    phoebe_direc = os.path.split(phoebe_basename)[0]
    if not os.path.isdir(phoebe_direc):
        os.mkdir(phoebe_direc)    
    
    #-- we're not going to use the mesh density from Phoebe Legacy, we're
    #   not quite there yet! Lower the sampling
    if mesh=='marching':
        system[0].params['mesh']['delta'] = delta # 0.06
        system[1].params['mesh']['delta'] = delta
    else:
        system[0].params['mesh']['gridsize'] = 20
        system[1].params['mesh']['gridsize'] = 20
    
    #-- Use Phoebe 2.0 to compute the light curve in the same time range, but we
    #   override the number of time points to use (Phoebe Legacy is way faster
    #   remember). Make sure to conserve volume at periastron: we compute the system at
    #   T0 (time of periastron passage). Phoebe 2.0 remembers the volume at the
    #   first computed time point
    time = np.hstack([phoebe_crit_times[cvol_index]-P/2,np.linspace(otime[0],otime[-1],250)])
    
    #--------- WD -----------------------------------------------------------
    #-- read in the WD lcin file, and make sure we compute stuff in JD
    wd_input_file = legacy_basename + '-test.lcin'
    ps,lc,rv = wd.lcin_to_ps(wd_input_file,version='wdphoebe')
    lc['jdstrt'] = time[1]
    lc['jdend'] = time[-1]+time[-1]-time[-2]
    lc['jdinc'] = time[-1]-time[-2]
    lc['indep_type'] = 'time (hjd)'
   
    #-- then create a BodyBag from WD: we want to make sure the output here
    #   is the same as before
    comp1,comp2,binary = wd.wd_to_phoebe(ps,lc,rv)
    star1,lcdep1,rvdep1 = comp1
    star2,lcdep2,rvdep2 = comp2
    mesh1 = phoebe.ParameterSet(frame='phoebe',context='mesh:marching',delta=0.2,alg='c')
    mesh2 = phoebe.ParameterSet(frame='phoebe',context='mesh:marching',delta=0.2,alg='c')
    star1 = phoebe.BinaryRocheStar(star1,binary,mesh1,pbdep=[lcdep1,rvdep1])
    star2 = phoebe.BinaryRocheStar(star2,binary,mesh2,pbdep=[lcdep2,rvdep2])
    wd_bbag = phoebe.BodyBag([star1,star2])
    
    #-- so write a file to compare the two (that's up to you...)
    curve,params = wd.lc(ps,request='curve',light_curve=lc,rv_curve=rv)
    serial_legacy = universe.serialize(system,color=False)
    serial_wildev = universe.serialize(wd_bbag,color=False)
    with open(phoebe_out_file,'w') as ff:
        for line1,line2 in zip(serial_legacy.split('\n'),serial_wildev.split('\n')):
            ff.write('PH:'+line1+'\n')
            ff.write('WD:'+line2+'\n')
    #============ COMPUTATIONS ===============================================
    #-- get mpi-stuff and details, but only if we're not profiling the code.
    if 'cProfile' in globals():
        mpi = None
    else:
        mpi = get_mpi_parameters()

    #-- compute the system if the light curves haven't been computed before
    if not os.path.isfile(phoebe_lc_file) or recompute:
        #-- compute the system
        phoebe.observe(system,time,lc=True,mpi=mpi,subdiv_num=0,refl=False,heating=True)
        #-- retrieve the results: for each component and for the whole system
        lcref = system[0].params['pbdep']['lcdep'].values()[0]['ref']
        lc = system.get_synthetic(type='lcsyn',ref=lcref)
        lc1 = system[0].get_synthetic(type='lcsyn',ref=lcref)
        lc2 = system[1].get_synthetic(type='lcsyn',ref=lcref)
        
        lc.save(phoebe_lc_file)
        lc1.save(phoebe_lc1_file)
        lc2.save(phoebe_lc2_file)
        time,flux = np.array(lc['time'][1:]),np.array(lc['flux'][1:])
        flux1 = np.array(lc1['flux'])[1:]
        flux2 = np.array(lc2['flux'])[1:]
    #-- if the system is already computed, just load the previous results.
    else:
        time,flux,sigma = np.loadtxt(phoebe_lc_file)[1:].T
        time1,flux1,sigma = np.loadtxt(phoebe_lc1_file)[1:].T
        time2,flux2,sigma = np.loadtxt(phoebe_lc2_file)[1:].T
        
    
    
    print "Flux ratio",(flux1/flux2).mean(),(flux1/flux2).std()
    
    #-- make a comparison plot
    plt.figure()
    plt.axes([0.1,0.3,0.85,0.6])
    plt.plot(time,flux/flux.mean(),'ko-',label='Phoebe 2.0')
    plt.plot(otime,oflux/oflux.mean(),'r-',lw=2,label='Phoebe Legacy')
    plt.plot(curve['indeps'],curve['lc']/curve['lc'].mean(),'bo-',lw=2,label='WD')
    plt.legend(loc='best').get_frame().set_alpha(0.5)
    plt.xlim(time.min(),time.max())
    plt.axes([0.1,0.05,0.85,0.2])
    plt.plot(time,(flux/flux.mean()-curve['lc']/curve['lc'].mean())*100.,'ko-')
    plt.ylabel('Residuals [%]')
    plt.xlim(time.min(),time.max())
    plt.figure()
    plt.plot(time,flux,'ko-',label='Total light curve')
    plt.plot(time,flux1,'ro-',label='Primary component')
    plt.plot(time,flux2,'bo-',label='Secondary component')
    plt.xlim(time.min(),time.max())
    plt.legend(loc='best').get_frame().set_alpha(0.5)
    
    #-- meshes:
    system[0].params['orbit']['phshift'] = 0.
    for i,(crit_time,crit_file) in enumerate(zip(legacy_crit_times,legacy_crit_files)):
        # quick hack:
        crit_time = phoebe_crit_times[i]
        #-- read in the Phoebe Legacy mesh:
        V,W = np.loadtxt(crit_file).T
        #-- compute the Phoebe 2.0 mesh:
        if not os.path.isfile(phoebe_crit_files[i]):
            system.set_time(crit_time)
            #-- only keep the visible ones
            eclipse.convex_bodies(system.get_bodies())
            visible = system.mesh['visible']
            X,Y = system.mesh['center'][visible,0],system.mesh['center'][visible,1]
            sma = system[0].params['orbit']['sma']
            X = X/sma
            Y = Y/sma
            np.savetxt(phoebe_crit_files[i],np.column_stack([X,Y]))
            if i==0:
                print(system[0].as_point_source())
                print(system[1].as_point_source())

        else:
            X,Y = np.loadtxt(phoebe_crit_files[i]).T
        
        #-- make the plots
        plt.figure()
        plt.subplot(111,aspect='equal')
        plt.title(crit_file)
        plt.plot(V,W,'ko',ms=7)
        plt.plot(X,Y,'ro',mec='r',ms=5)
        plt.grid()
        
        
    
    
    

def get_mpi_parameters():
    #-- setup MPI stuff: check which host the script is running on. If it is
    #   clusty and we can readily find a hosts file, we fill in the MPI parameter
    #   set.
    hostname = socket.gethostname()
    if hostname=='clusty':
        hostfile = os.path.expanduser('~/mpi/hosts')
        print("Running on Clusty, trying to load hostfile {}".format(hostfile)) 
        if not os.path.isfile(hostfile):
            print("Cannot load hostfile {} (file does not exist), falling back to defaults".format(hostfile))
        else:
            mpi = phoebe.ParameterSet(context='mpi',np=24,hostfile=hostfile,
                                        byslot=True,python='python2.7')
    #-- If the hostname isn't clusty, we're probably running on a normal computer
    #   in that case, just take the number of available processes.
    else:
        mpi = phoebe.ParameterSet(context='mpi',np=multiprocessing.cpu_count())
    return mpi 


def get_meshes(name,type=''):
    #-- Retrieve the filenames and the times from the meshes
    times_file = name+'-important_times.dat'
    phasenames = ['mesh-periastron','mesh-sup_conj','mesh-inf_conj',
                  'mesh-asc_node','mesh-desc_node']
    #-- read in the file with the times
    with open(times_file,'r') as ff:
        crit_times = [float(line.strip().split()[-1]) for line in ff.readlines()[2:]]
    #-- output the filenames with the times
    crit_files = [name+'-'+phasename+type+'.dat' for phasename in phasenames]
    return crit_times,crit_files
            
    

if __name__=="__main__":
    
    #-- collect command line information
    recompute = False
    if len(sys.argv[1:]):
        recompute = ('--recompute' in sys.argv[1:])
        test_cases = sys.argv[1:]
    else:
        test_cases = sorted(glob.glob('*'))
    if recompute:
        test_cases.pop(test_cases.index('--recompute'))
    compare(test_cases[0],recompute=recompute)
    
    plt.show()