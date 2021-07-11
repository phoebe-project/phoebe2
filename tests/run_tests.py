"""
General script to run all tests before committing

pylint: sudo pip install pylint

"""

import matplotlib
matplotlib.use('Agg')

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess

if len(sys.argv)==1:
    do = ['tutorials', 'doctests', 'nosetests', 'pylint', 'benchmark']
else:
    do = sys.argv

cwd = os.getcwd()

if 'tutorials' in do:
    import ipynbtest
    # os.chdir(os.path.join(cwd, '../docs/tutorials'))
    if True:
        print("TESTING TUTORIALS...")
        ipynbtest.test_all_in_dir(os.path.join(cwd, '../docs/tutorials'), interactive='interactive' in do)
    if True:
        print("TESTING EXAMPLE SCRIPTS...")
        ipynbtest.test_all_in_dir(os.path.join(cwd, '../docs/examples'), interactive='interactive' in do)


if 'nosetests' in do:
    print("TESTING NOSETESTS... (no output until finished or failed)")
    os.chdir(os.path.join(cwd, 'nosetests'))
    # import nose
    # nose.run()
    out = subprocess.check_output('nosetests -xv', shell=True, stderr=subprocess.DEVNULL).decode('utf-8').strip()
    if 'FAILED' in out:
        print("NOSETESTS FAILED: \n{}".format(out))
        exit()

# TODO: run style checks
if 'pylint' in do:
    print("TESTING PYTHON STYLE VIA PYLINT...")
    print("TODO: support style checks")
    out = subprocess.check_output('pylint phoebe', shell=True, stderr=subprocess.DEVNULL).decode('utf-8').strip()

# TODO: run API doctests
if 'doctests' in do:
    print("TESTING API VIA DOCTESTS...")
    print("TODO: support doctests")

if 'benchmark' in do or 'benchmarks' in do:
    print("RUNNING BENCHMARKS...")

    branch_name =  subprocess.check_output('git rev-parse --symbolic-full-name --abbrev-ref HEAD', shell=True, stderr=subprocess.DEVNULL).decode('utf-8').strip()
    commit_hash = subprocess.check_output('git log -n 1 --pretty=format:"%H"', shell=True, stderr=subprocess.DEVNULL).decode('utf-8').strip()

    print("   branch: {}, commit: {}".format(branch_name, commit_hash))

    os.chdir(os.path.join(cwd, 'benchmark'))
    times = {}
    for fname in glob('./*py'):
        f_py =  os.path.basename(fname)
        f_profile = f_py.split('.py')[0]+'.profile'
        print("running {} to create {}".format(f_py, f_profile))

        out = subprocess.check_output('time python {}'.format(f_py), shell=True, stderr=subprocess.STDOUT).decode('utf-8')
        times[f_py] = float(out.split('user')[0])
        print("   {} s".format(times[f_py]))

        f_result_fname = f_py.split('.py')[0]+'.log'
        f_result = open(f_result_fname, 'a')
        f_result.write("{} {} {}\n".format(branch_name, commit_hash, times[f_py]))
        f_result.close()

        print("plotting benchmark history for {}...".format(f_py))
        f_result = open(f_result_fname, 'r')
        lines = f_result.readlines()
        f_result.close()

        branches_plot = ['master', 'development']
        branches = {branch: [] for branch in branches_plot}

        for line in lines:
            branch, commit, time = line.strip().split()
            for branch_i in branches_plot:
                if branch_i == branch:
                    branches[branch].append(time)
                elif branch in branches.keys():
                    branches[branch_i].append(np.nan)
                else:
                    continue

        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)
        for branch, benchmark_ts in branches.items():
            ax.plot(range(len(benchmark_ts)), benchmark_ts, 'o', label=branch)
        plt.legend()
        ax.set_title(f_py.split('.py')[0])
        ax.set_xlabel('commit')
        ax.set_ylabel('benchmark time (s)')
        fig.savefig(f_result_fname+'.png')



    print("PROFILE TIMES (see individual .profile files for details)")
    for f_py, time in times.items():
        print("{}: {}".format(f_py, time))
