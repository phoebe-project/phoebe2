"""
General script to run all tests before committing

pylint: sudo pip install pylint

"""

from glob import glob
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
    os.chdir(os.path.join(cwd, 'benchmark'))
    times = {}
    for fname in glob('./*py'):
        f_py =  os.path.basename(fname)
        f_profile = f_py.split('.py')[0]+'.profile'
        print "running {} to create {}".format(f_py, f_profile)

        out  = commands.getoutput('time python -m cProfile -o {} {}'.format(f_profile, f_py))
        times[f_py] = float(out.split()[-9].split('user')[0])


    print "PROFILE TIMES (see individual .profile files for details)"
    for f_py, time in times.items():
        print "{}: {}".format(f_py, time)
