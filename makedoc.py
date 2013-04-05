"""
Automatically generate full source documentation and tutorial web pages.

Optionally, also copy the generated pages to the dedicated website.
"""
import os
import shutil
import glob
import sys
import subprocess
import time
import datetime
from phoebe.parameters import definitions
from phoebe.parameters import parameters

def python_to_sphinx(pythonfile,latex=False,type='testsuite.'):
    """
    Convert a Python script (*.py) to a Sphinx file (*.rst).
    """
    myname = type+os.path.splitext(os.path.basename(pythonfile))[0]
    
    ff = open(pythonfile,'r')
    tt = open('phoebe-doc/{}.rst'.format(myname),'w')

    start_doc = False
    inside_code = False

    for line in ff.readlines():
        if 'time.time(' in line: continue
        if 'import time' in line: continue
        if 'os.system' in line: continue
        if 'subprocess.call' in line: continue
        if '***time***' in line: line = line.replace('***time***','{}'.format(str(datetime.datetime.today())))
        if latex and '.gif' in line:
            line = line.replace('.gif','.png')
        if latex and '.svg' in line:
            line = line.replace('.svg','.png')
            
        if not start_doc and line[:3]=='"""':
            start_doc = True
            if inside_code:
                tt.write('\n')
                inside_code = False
            continue
        
        if start_doc and line[:3]=='"""':
            start_doc = False
            if inside_code:
                tt.write('\n')
                inside_code = False
            continue
        
        if start_doc:
            tt.write(line)
            if inside_code:
                tt.write('\n')
                inside_code = False
            continue
        
        if line[0]=='#':
            if inside_code:
                tt.write('\n')
                inside_code = False
            tt.write(line[1:].strip()+'\n')
            continue
        
        if not inside_code and not line.strip():
            continue
        
        if not inside_code:
            inside_code = True
            tt.write('\n::\n\n')
        
        tt.write('    '+line)
    ff.close()
    tt.close()

def generate_parameterlist_sphinx():
    """
    Generate a list of parameters suitable for inclusion in sphinx.
    """
    with open('phoebe-doc/parlist.rst','w') as ff:
        ff.write("""

.. _list-of-parameters:        
        
List of ParameterSets and Parameters
=============================================

ParameterSets
-------------

""")
        #-- keep track of all frames and contexts.
        frames = {}
        for par in definitions.defs:
            for frame in par['frame']:
                if not frame in ['phoebe','pywd']: continue
                if frame not in frames:
                    if isinstance(par['context'],list):
                        frames[frame]+= par['context']
                    else:
                        frames[frame] = [par['context']]
                elif not par['context'] in frames[frame]:
                    if isinstance(par['context'],list):
                        frames[frame]+= par['context']
                    else:
                        frames[frame].append(par['context'])

        frames_contexts = []
        for frame in sorted(frames.keys()):
            for context in sorted(frames[frame]):
                if frame+context in frames_contexts: continue
                frames_contexts.append(frame+context)
                parset = parameters.ParameterSet(frame=frame,context=context)
                if 'label' in parset:
                    parset['label'] = 'mylbl'
                if 'ref' in parset:
                    parset['ref'] = 'myref'
                if 'c1label' in parset:
                    parset['c1label'] = 'primlbl'
                if 'c2label' in parset:
                    parset['c2label'] = 'secnlbl'
                
                
                str_parset = str(parset).split('\n')
                if len(str_parset)<2: continue
                #-- add a label
                ff.write('.. _parlabel-{}-{}:\n\n'.format(frame,context))
                #-- add the parameterset
                ff.write('**{}** ({})::\n\n'.format(context,frame))
                str_parset = '    '+'\n    '.join(str_parset)                
                ff.write(str_parset)
                ff.write('\n\n')



if __name__=="__main__":
    
    generate_parameterlist_sphinx()
    giffiles = sorted(glob.glob('phoebe-doc/images_tut/*.gif'))
    for giffile in giffiles:
        subprocess.call('convert {}[0] {}'.format(giffile,os.path.splitext(giffile)[0]+'.png'),shell=True)
    
    test_suite = ['phoebe-testsuite/solar_calibration/solar_calibration.py',
                  'phoebe-testsuite/vega/vega.py',
                  'phoebe-testsuite/vega/vega_sed.py',
                  'phoebe-testsuite/sirius/sirius.py',
                  'phoebe-testsuite/wilson_devinney/wd_vs_phoebe.py',
                  'phoebe-testsuite/wilson_devinney/eccentric_orbit.py',
                  'phoebe-testsuite/wilson_devinney/reflection_effect.py',
                  'phoebe-testsuite/venus/venus.py',
                  'phoebe-testsuite/differential_rotation/differential_rotation.py',
                  'phoebe-testsuite/fast_rotator/fast_rotator.py',
                  'phoebe-testsuite/critical_rotator/critical_rotator.py',
                  'phoebe-testsuite/spotted_star/spotted_star.py',
                  'phoebe-testsuite/pulsating_star/pulsating_star.py',
                  'phoebe-testsuite/pulsating_binary/pulsating_binary.py',
                  'phoebe-testsuite/pulsating_binary/pulsating_binary2.py',
                  'phoebe-testsuite/pulsating_rotating/pulsating_rotating.py',
                  'phoebe-testsuite/beaming/KPD1946+4340.py',
                  'phoebe-testsuite/example_systems/example_systems.py',
                  'phoebe-testsuite/occulting_dark_sphere/occulting_dark_sphere.py',
                  'phoebe-testsuite/occulting_dark_sphere/transit_colors.py']
    
    for pythonfile in test_suite:
        python_to_sphinx(pythonfile,type='testsuite.',latex=False)
    python_to_sphinx('phoebe-doc/scripts/how_to_binary.py',type='',latex=False)
    
    subprocess.call('sphinx-apidoc -f -o phoebe-doc phoebe',shell=True)
    os.chdir('phoebe-doc')
    subprocess.call('make html',shell=True)
    
    if 'pdf' in sys.argv[1:]:
        os.chdir('..')
        for pythonfile in test_suite:
            python_to_sphinx(pythonfile,type='testsuite.',latex=True)
        os.chdir('phoebe-doc')
        subprocess.call('make latexpdf',shell=True)
        shutil.copy('_build/latex/phoebe.pdf','_build/html/phoebe.pdf')
    
    if 'copy' in sys.argv[1:]:
         #subprocess.call('scp -r _build/html/* copernicus.ster.kuleuven.be:public_html/phoebe_alt',shell=True)
         subprocess.call('scp -r _build/html/* clusty.ast.villanova.edu:/srv/www/phoebe/2.0/docs/',shell=True)