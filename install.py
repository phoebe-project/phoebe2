"""
Install Phoebe and all its dependencies in a virtual environment.

To install, simply do one of::

    $:> python install.py MYDIR
    $:> python install.py --auto-update MYDIR
    
With ``MYDIR`` the directory where Phoebe needs to be installed to. ``MYDIR``
defaults to the directory ``phoebe`` in the current working directory. Make sure
the directory does not exist.

To update, simply do the same as for the installation::

    $:> python install.py MYDIR
    
If you add the option ``--auto-update`` during install or update, the
installation will be automatically updated at midnight every day. In that case,
make sure the ``install.py`` file remains at the same location. Removing the
cronjob can be done via::

    $:> crontab -e
    
and following the instructions.
    
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

if '--auto-update' in sys.argv[1:]:
    add_cronjob = tuple(list(sys.argv).pop(sys.argv.index('--auto-update')))
else:
    add_cronjob = False
        
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
    urllib.urlretrieve('https://raw.github.com/pypa/virtualenv/master/virtualenv.py', "virtualenv.py")
    #urllib.urlretrieve('https://bitbucket.org/ianb/virtualenv/raw/tip/virtualenv.py', "virtualenv.py")

    #-- install the virtualenv    
    import virtualenv
    virtualenv.main()

#-- download the requirement files
req_files = ['numpy-basic.txt','phoebe-basic.txt','phoebe-full.txt']
for req_file in req_files:
    if not os.path.isfile(req_file):
        print("Downloaded requirement file {}".format(req_file))
        urllib.urlretrieve('http://www.phoebe-project.org/2.0/docs/_downloads/{}'.format(req_file), req_file)

#-- install the basic requirements
activate = '. '+os.path.join(base_dir,'bin','activate')
things_to_do = ['pip','install','-r','numpy-basic.txt'],\
               ['pip','install','-r','phoebe-basic.txt'],\
               ['pip','install','-r','phoebe-full.txt'],
fail = False

new_package = False
nr = 0
last_char_was_dot = False
with open('install.log','w') as log:
    for thing_to_do in things_to_do:
        cmd = ' '.join([activate]+['&&']+thing_to_do)
        p1 = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True)
        print("Running {}".format(cmd))
        for line in line_at_a_time(p1.stdout):
            log.write(line)
            if 'Requirement already satisfied' in line:
                if last_char_was_dot:
                    print('')
                    last_char_was_dot = False
                print(line.strip())
            elif 'Successfully installed' in line:
                new_package = False
                sys.stdout.write('done.\n')
                sys.stdout.flush()
                print(line.strip())
                last_char_was_dot = False
            elif "Downloading/unpacking" in line or 'Obtaining' in line:
                new_package = True
                line = line.split()
                if last_char_was_dot:
                    print('')
                    last_char_was_dot = False
                sys.stdout.write(' '.join(line[:1]+['/installing']+line[1:]))
                sys.stdout.flush()
            elif new_package:
                nr += 1
                if nr%100==0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    last_char_was_dot = True
        if last_char_was_dot:
            print('')
            last_char_was_dot = False
                
        log.flush()

#-- finally download the atmosphere files:
files = ['kurucz_p00_claret_equidist_r_leastsq_teff_logg.fits',
         'blackbody_uniform_none_teff.fits']

for ff in files:
    source = 'http://www.phoebe-project.org/2.0/docs/_downloads/'+ff
    destin = os.path.join(base_dir,'src/phoebe/phoebe/atmospheres/tables/ld_coeffs/'+ff)
    if not os.path.isfile(destin):
        urllib.urlretrieve(source,destin)
        print("Added table {}".format(destin))
    else:
        print("Table {} already exists".format(destin))

#-- if you want a Cron job; do that here:
if add_cronjob:
    cronjob = "0 * * * * python {} {}".format(os.path.abspath(sys.argv[0]),base_dir)
    flag = subprocess.call('(crontab -l ; echo "{}") |uniq - | crontab -'.format(cronjob),shell=True)
    if not flag:
        print("Succesfully added cronjob '{}'".format(cronjob))
    else:
        print("Could not add cronjob")
else:
    print("Did not add cronjob for regular updating")

print("Don't forget to source the virtual environment before using Phoebe:\nsource {}".format(os.path.join(base_dir,'bin/activate')))

