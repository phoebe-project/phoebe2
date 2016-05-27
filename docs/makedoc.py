"""
requires sphinx (sudo apt-get install python-sphinx)
requires runipy (pip install runipy)
requires ipython (sudo pip install jupyter; sudo apt-get install ipython-notebook)
requires pandoc (sudo apt-get install pandoc)
"""


from glob import glob
import os
import sys
import datetime


if __name__ == '__main__':

    cwd = os.getcwd()

    if 'skiptests' not in sys.argv:
        print "TESTING TUTORIALS"
        sys.path.append(os.path.join(cwd, '../tests'))
        import ipynbtest
        #os.chdir(os.path.join(cwd, 'tutorials'))
        ipynbtest.test_all_in_dir(os.path.join(cwd, 'tutorials'))

        print "TESTING EXAMPLE SCRIPTS"
        ipynbtest.test_all_in_dir(os.path.join(cwd, 'examples'))

    print "CONVERTING TUTORIALS TO RST AND PY FILES"
    os.chdir(os.path.join(cwd, 'tutorials'))
    os.system('ipython nbconvert --to rst *ipynb')
    os.system('ipython nbconvert --to python *ipynb')

    print "CONVERTING EXAMPLES SCRIPTS TO RST AND PY FILES"
    os.chdir(os.path.join(cwd, 'examples'))
    os.system('ipython nbconvert --to rst *ipynb')
    os.system('ipython nbconvert --to python *ipynb')

    print "BUILDING API"
    os.chdir(cwd)
    os.system('sphinx-apidoc -f -o api/ ../phoebe2/')

    print "BUILDING HTML"
    os.system('make html')
