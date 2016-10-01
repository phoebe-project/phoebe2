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
import commands

if len(sys.argv)==1:
    do = ['tutorials', 'doctests', 'nosetests', 'pylint', 'benchmark']
else:
    do = sys.argv

cwd = os.getcwd()

if 'tutorials' in do:
    import ipynbtest
    # os.chdir(os.path.join(cwd, '../docs/tutorials'))
    if True:
        print "TESTING TUTORIALS..."
        ipynbtest.test_all_in_dir(os.path.join(cwd, '../docs/tutorials'), interactive='interactive' in do)
    if True:
        print "TESTING EXAMPLE SCRIPTS..."
        ipynbtest.test_all_in_dir(os.path.join(cwd, '../docs/examples'), interactive='interactive' in do)


if 'nosetests' in do:
    print "TESTING NOSETESTS... (no output until finished or failed)"
    os.chdir(os.path.join(cwd, 'nosetests'))
    # import nose
    # nose.run()
    out = commands.getoutput('nosetests -xv')
    if 'FAILED' in out:
        print "NOSETESTS FAILED: \n{}".format(out)
        exit()

# TODO: run style checks
if 'pylint' in do:
    print "TESTING PYTHON STYLE VIA PYLINT..."
    print "TODO: support style checks"
    out = commands.getoutput('pylint phoebe')

# TODO: run API doctests
if 'doctests' in do:
    print "TESTING API VIA DOCTESTS..."
    print "TODO: support doctests"

if 'benchmark' in do or 'benchmarks' in do:
    print "RUNNING BENCHMARKS..."

    branch_name =  os.environ.get('TRAVIS_BRANCH', commands.getoutput('git rev-parse --symbolic-full-name --abbrev-ref HEAD'))
    commit_hash = os.environ.get('TRAVIS_COMMIT', commands.getoutput('git log -n 1 --pretty=format:"%H"'))

    os.chdir(os.path.join(cwd, 'benchmark'))
    times = {}
    for fname in glob('./*py'):
        f_py =  os.path.basename(fname)
        f_profile = f_py.split('.py')[0]+'.profile'
        print "running {} to create {}".format(f_py, f_profile)

        out  = commands.getoutput('time python -m cProfile -o {} {}'.format(f_profile, f_py))
        times[f_py] = float(out.split()[-9].split('user')[0])



        f_result_fname = f_py.split('.py')[0]+'.log'
        f_result = open(f_result_fname, 'a')
        f_result.write("{} {} {}\n".format(branch_name, commit_hash, times[f_py]))
        f_result.close()

        print "plotting benchmark history for {}...".format(f_py)
        f_result = open(f_result_fname, 'r')
        lines = f_result.readlines()
        f_result.close()

        branches_skip = []
        branches = {}
        for line in lines:
            branch, commit, time = line.strip().split()
            if branch not in branches.keys() and branch not in branches_skip:
                branches[branch] = []

        for line in lines:
            branch, commit, time = line.strip().split()
            for branch_i in branches.keys():
                if branch_i == branch:
                    branches[branch].append(time)
                else:
                    branches[branch].append(np.nan)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = range(len(lines))
        for branch, benchmark_ts in branches.items():
            ax.plot(x, benchmark_ts, 'o-', label=branch)
        plt.legend()
        ax.set_title(f_py.split('.py')[0])
        ax.set_xlabel('commit')
        ax.set_ylabel('benchmark time (s)')
        fig.savefig(f_result_fname+'.png')



    print "PROFILE TIMES (see individual .profile files for details)"
    for f_py, time in times.items():
        print "{}: {}".format(f_py, time)

