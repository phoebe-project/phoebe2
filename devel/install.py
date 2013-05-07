"""
Install Phoebe and all its dependencies in a virtual environment.

To install, simply do::

    $:> python install.py MYDIR
    
With ``MYDIR`` the directory where Phoebe needs to be installed to. ``MYDIR``
defaults to the directory ``phoebe`` in the current working directory. Make sure
the directory does not exist.

To update, simply do the same as for the installation::

    $:> python install.py MYDIR
    
To unistall, simply remove the directory::

    $:> rm -rf MYDIR
    
"""

import urllib
import sys
import os
import subprocess

class InstallationError(Exception):
    def __init__(self,message):
        # Call the base class constructor with the parameters it needs
        self.message = message
    def __str__(self):
        return self.message
    
def line_at_a_time(fileobj):
    while True:
        line = fileobj.readline()
        if not line:
            return
        yield line
        
if len(sys.argv)==1:
    sys.argv.append('phoebe')

base_dir = sys.argv[-1]

#-- make sure the virtualenv starts with a clean slate
if not '--no-site-packages' in sys.argv:
    sys.argv.append('--no-site-packages')

if os.path.isdir(base_dir):
    print("Virtual environment already exists")
else:
    #-- download the virtualenv.py script
    urllib.urlretrieve('https://bitbucket.org/ianb/virtualenv/raw/tip/virtualenv.py', "virtualenv.py")

    #-- install the virtualenv    
    import virtualenv
    virtualenv.main()

#-- download the requirement files
urllib.urlretrieve('http://www.phoebe-project.org/2.0/docs/_downloads/numpy-basic.txt', "numpy-basic.txt")
urllib.urlretrieve('http://www.phoebe-project.org/2.0/docs/_downloads/phoebe-basic.txt', "phoebe-basic.txt")
urllib.urlretrieve('http://www.phoebe-project.org/2.0/docs/_downloads/phoebe-full.txt', "phoebe-full.txt")

#-- install the basic requirements
activate = '. '+os.path.join(base_dir,'bin','activate')
things_to_do = ['pip','install','-r','numpy-basic.txt'],\
               ['pip','install','-r','phoebe-basic.txt'],\
               ['pip','install','-r','phoebe-full.txt'],
fail = False

new_package = False
nr = 0
with open('install.log','w') as log:
    for thing_to_do in things_to_do:
        cmd = ' '.join([activate]+['&&']+thing_to_do)
        p1 = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True)
        print("Running {}".format(cmd))
        for line in line_at_a_time(p1.stdout):
            log.write(line)
            if 'Requirement already satisfied' in line:
                print('')
                print(line.strip())
            elif 'Successfully installed' in line:
                new_package = False
                sys.stdout.write('done.\n')
                sys.stdout.flush()
                print(line.strip())
            elif "Downloading/unpacking" in line or 'Obtaining' in line:
                new_package = True
                line = line.split()
                print('')
                sys.stdout.write(' '.join(line[:1]+['/installing']+line[1:]))
                sys.stdout.flush()
            elif new_package:
                nr += 1
                if nr%100==0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
        print('')
        log.flush()

#-- finally download the atmosphere files:
files = ['kurucz_p00_claret_equidist_r_leastsq_teff_logg.fits',
         'blackbody_uniform_none_teff.fits']

for ff in files:
    source = 'http://www.phoebe-project.org/2.0/docs/_downloads/'+ff
    destin = os.path.join(basedir,'src/phoebe/atmospheres/tables/ld_coeffs/'+ff)
    urllib.urlretrieve(source,destin)
